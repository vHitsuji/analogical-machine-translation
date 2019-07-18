import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
from progressbar import progressbar
from math import sqrt


matrix_length = 13
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

        self.layers_size = [8*self.matrix_size,8*self.matrix_size]

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
                x = F.relu(self.channels[channel_index][layer_index](x))
            x = torch.tanh(self.channels[channel_index][-1](x))
            output_list.append(x)

        m6, m7, m8 = output_list

        result = torch.cat((m6, m7, m8), 1)
        return result

class simplePixelDense(nn.Module):

    def __init__(self, matrix_length):
        super(simplePixelDense, self).__init__()

        self.matrix_length = matrix_length
        self.matrix_size = self.matrix_length * self.matrix_length

        self.layers_size = [5*self.matrix_size, int(sqrt(5*self.matrix_size))+1]

        # Initiates layers in 3 channels
        self.channels = list()
        for channel_index in range(3):
            channel = list()
            channel.append(nn.Linear(5 * self.matrix_size, self.layers_size[0]))
            for layer_index in range(len(self.layers_size)-1):
                channel.append(nn.Linear(self.layers_size[layer_index], self.layers_size[layer_index+1]))
            channel.append(nn.Linear(self.layers_size[-1], 1))

            self.channels.append(nn.ModuleList(channel))
        self.channels = nn.ModuleList(self.channels)



    def forward(self, batch):
        m0123 = batch[:, [0, 1, 2, 3, 6], :, :].view(-1, 5 * self.matrix_size) #Last matrix is pixel position encoding
        m2345 = batch[:, [2, 3, 4, 5, 6], :, :].view(-1, 5 * self.matrix_size)
        m0154 = batch[:, [0, 1, 5, 4, 6], :, :].view(-1, 5 * self.matrix_size)

        input_list = [m0123, m2345, m0154]
        output_list = list()
        for channel_index in range(3):
            x = input_list[channel_index]
            for layer_index in range(len(self.channels[channel_index])-1):
                x = F.relu(self.channels[channel_index][layer_index](x))
            x = torch.tanh(self.channels[channel_index][-1](x))
            output_list.append(x)

        m6, m7, m8 = output_list

        result = torch.cat((m6, m7, m8), 1)
        return result


class cuboidDataset(data.Dataset):
    def __init__(self, matrices_file_path, pixel_mode=False):
        dataset = np.load(matrices_file_path)

        self.__pixel_mode = pixel_mode

        self.__analogies_index = dataset["analogies"]
        self.__sentence_couples_index = dataset["index"]
        self.__matrices = dataset["matrices"]
        self.__matrix_length = self.__matrices.shape[1]
        self.__matrix_size = self.__matrices.shape[1]*self.__matrices.shape[2]
        assert(self.__matrix_size == 13*13)

        if self.__pixel_mode:
            self.__length = len(self.__analogies_index)*self.__matrix_size
            self.__position_encodings = list()
            for i in range(self.__matrix_length):
                for j in range(self.__matrix_length):
                    matrix = (-1)*np.ones((self.__matrix_length, self.__matrix_length), dtype=np.float)
                    matrix[i, j] = 1
                    self.__position_encodings.append(matrix)
            self.__position_encodings = torch.from_numpy(np.array(self.__position_encodings)).float()
        else:
            self.__length = len(self.__analogies_index)

        self.__analogies_lengths = dataset["lengths"]

        self.__matrices = torch.from_numpy(self.__matrices)
        self.__matrices = self.__matrices.float()
        self.__analogies_lengths = torch.from_numpy(self.__analogies_lengths)
        self.__analogies_lengths = self.__analogies_lengths.float()


    def __len__(self):
        return self.__length

    def __getitem__(self, index):

        if self.__pixel_mode:
            analogy_index, pixel_index = divmod(index, self.__matrix_size)
            x_index, y_index = divmod(pixel_index, self.__matrix_length)

            matrices_indices = self.__analogies_index[analogy_index]

            input_example = torch.cat(
                (
                    self.__matrices[matrices_indices[:6], :, :],
                    torch.unsqueeze(self.__position_encodings[pixel_index], 0)
                ),
                0)
            truth = self.__matrices[matrices_indices[6:], x_index, y_index]

            return input_example, truth


        else:
            matrices_indices = self.__analogies_index[index]
            return self.__matrices[matrices_indices[:6], :, :], \
                self.__matrices[matrices_indices[6:], :, :]





def train_matrices():
    print("Is cuda available?", torch.cuda.is_available() )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 4}
    max_epochs = 10

    training_set = cuboidDataset("en-fr.matrices.npz", pixel_mode=True)
    training_generator = data.DataLoader(training_set, **params)

    net = simplePixelDense(matrix_length).to(device)
    print("Net architecture:")
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction="none")


    print_frequency = 100

    for epoch in range(max_epochs):

        running_loss = 0.0
        # Training
        for i, (local_batch, local_truths) in enumerate(training_generator):

            local_batch, local_truths = local_batch.to(device), local_truths.to(device)

            optimizer.zero_grad()
            outputs = net(local_batch)
            squares = criterion(outputs, local_truths.view(-1, 3))
            #squares = criterion(outputs, local_truths.view(-1, 3 * matrix_size))
            loss = torch.mean(squares)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_frequency == print_frequency-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / print_frequency))
                running_loss = 0.0
                print("max: ", torch.max(squares).item())



train_matrices()
