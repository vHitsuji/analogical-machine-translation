import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data


class simpleLinear(nn.Module):

    def __init__(self):
        super(simpleLinear, self).__init__()

        self.matrix_length = 32
        self.matrix_size = self.matrix_length * self.matrix_length

        self.bilingual = nn.Linear(4 * self.matrix_size, self.matrix_size)
        self.monolingual = nn.Linear(4 * self.matrix_size, self.matrix_size)

    def forward(self, x):
        m0123 = x[:, [0, 1, 2, 3], :, :].view(-1, 4 * self.matrix_size)
        m2345 = x[:, [2, 3, 4, 5], :, :].view(-1, 4 * self.matrix_size)
        m0154 = x[:, [0, 1, 5, 4], :, :].view(-1, 4 * self.matrix_size)
        m6 = self.bilingual(m0123)
        m7 = self.monolingual(m2345)
        m8 = self.monolingual(m0154)

        result = torch.cat((m6, m7, m8), 1)
        return result

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension (From Pytorch tutorial)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Dataset(data.Dataset):
    def __init__(self, nlg_file_path, matrices_file_path):
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        ID = self.list_IDs[index]

        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

net = simpleLinear()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


"""
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
"""