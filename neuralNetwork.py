import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
from progressbar import progressbar


matrix_length = 32
matrix_size = matrix_length*matrix_length

class simpleLinear(nn.Module):

    def __init__(self, matrix_length):
        super(simpleLinear, self).__init__()

        self.matrix_length = matrix_length
        self.matrix_size = self.matrix_length * self.matrix_length

        self.bilingual = nn.Linear(4 * self.matrix_size, self.matrix_size)
        self.monolingual1 = nn.Linear(4 * self.matrix_size, self.matrix_size)
        self.monolingual2 = nn.Linear(4 * self.matrix_size, self.matrix_size)

    def forward(self, x):
        m0123 = x[:, [0, 1, 2, 3], :, :].view(-1, 4 * self.matrix_size)
        m2345 = x[:, [2, 3, 4, 5], :, :].view(-1, 4 * self.matrix_size)
        m0154 = x[:, [0, 1, 5, 4], :, :].view(-1, 4 * self.matrix_size)
        m6 = self.bilingual(m0123)
        m7 = self.monolingual1(m2345)
        m8 = self.monolingual2(m0154)

        result = torch.cat((m6, m7, m8), 1)
        return result



class simpleDense(nn.Module):

    def __init__(self, matrix_length):
        super(simpleDense, self).__init__()

        self.matrix_length = matrix_length
        self.matrix_size = self.matrix_length * self.matrix_length

        self.layers_size = [8*self.matrix_size, 8*self.matrix_size,  4*self.matrix_size, 4*self.matrix_size]
        self.layers_activation = ["relu", "relu", "relu", "tanh"]

        # Initiates layers in 3 channels
        self.channels = list()
        for channel_index in range(3):
            channel = list()
            channel.append(nn.Linear(4 * self.matrix_size, self.layers_size[0]))
            for layer_index in range(len(self.layers_size)-1):
                channel.append(nn.Linear(self.layers_size[layer_index], self.layers_size[layer_index+1]))
            channel.append(nn.Linear(self.layers_size[-1], self.matrix_size))
            self.channels.append(nn.ModuleList(channel))
        self.channels = nn.ModuleList(self.channels)



    def forward(self, batch):
        m0123 = batch[:, [0, 1, 2, 3], :, :].view(-1, 4 * self.matrix_size)
        m2345 = batch[:, [2, 3, 4, 5], :, :].view(-1, 4 * self.matrix_size)
        m0154 = batch[:, [0, 1, 5, 4], :, :].view(-1, 4 * self.matrix_size)

        input_list = [m0123, m2345, m0154]
        output_list = list()
        for channel_index in range(3):
            x = input_list[channel_index]
            for layer_index in range(len(self.channels[channel_index])-1):
                if self.layers_activation[layer_index] == "tanh":
                    x = torch.tanh(self.channels[channel_index][layer_index](x))
                else:
                    x = F.relu(self.channels[channel_index][layer_index](x))
            x = self.channels[channel_index][-1](x)
            output_list.append(x)

        m6, m7, m8 = output_list

        result = torch.cat((m6, m7, m8), 1)
        return result



class cuboidDataset(data.Dataset):
    def __init__(self, matrices_file_path):
        dataset = np.load(matrices_file_path)
        self.__analogies_index = dataset["analogies"]
        self.__sentence_couples_index = dataset["index"]
        self.__matrices = dataset["matrices"]
        self.__length = len(self.__analogies_index)

        self.__matrices = torch.from_numpy(self.__matrices)
        self.__matrices = self.__matrices.float()



    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        matrices_indices = self.__analogies_index[index]

        return self.__matrices[matrices_indices[:6], :, :], \
            self.__matrices[matrices_indices[6:], :, :]  # (input, ground truth)



print("Is cuda available?", torch.cuda.is_available() )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 4}
max_epochs = 10

training_set = cuboidDataset("en-fr.matrices.npz")
training_generator = data.DataLoader(training_set, **params)

net = simpleDense(matrix_length).to(device)
print("Net architecture:")
print(net)
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


print_frequency = 10

for epoch in range(max_epochs):

    running_loss = 0.0
    # Training
    for i, (local_batch, local_truths) in enumerate(training_generator):

        local_batch, local_truths = local_batch.to(device), local_truths.to(device)

        optimizer.zero_grad()
        outputs = net(local_batch)
        loss = criterion(outputs, local_truths.view(-1, 3 * matrix_size))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_frequency == print_frequency-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / print_frequency))
            running_loss = 0.0


"""
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
"""