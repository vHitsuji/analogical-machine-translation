import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
from math import sqrt


def printMatrix(matrix, s1_words=None, s2_words=None):
    """
    Print an alignment matrix with given string as axis scale.
    Do not forget to plt.show() when you want to show the matrix.
    :param matrix: Matrix to show.
    :param s1_words: x_axis sentence.
    :param s2_words: y_axis sentence.
    :return: None
    """
    plt.matshow(matrix, cmap="gray_r")
    if s1_words is not None and s1_words is not None:
        plt.xticks(range(len(s2_words)), s2_words)
        plt.yticks(range(len(s1_words)), s1_words)


class simpleLinear(nn.Module):
    """ Always output one matrix"""
    def __init__(self, matrix_length, input_matrices=(0, 1, 2, 3, 4, 5)):
        super(simpleLinear, self).__init__()
        self.__input_matrices = input_matrices
        self.__matrix_length = matrix_length
        self.__matrix_size = self.__matrix_length * self.__matrix_length
        self.__input_size = len(input_matrices)
        self.__linearLayer = nn.Linear(self.__input_size * self.matrix_size, self.matrix_size)

    def forward(self, batch):
        batch = batch[:, self.__input_matrices, :, :].view(-1, self.__input_size * self.matrix_size)
        batch = self.__linearLayer(batch).view(-1, self.__matrix_length, self.__matrix_length)
        return batch


class simpleDense(nn.Module):

    def __init__(self, matrix_length, input_matrices=(0, 1, 2, 3, 4, 5), output_size=1, pixel_mode=False,
                 layers_size=None):
        super(simpleDense, self).__init__()
        self.__input_matrices = input_matrices
        self.__input_size = len(input_matrices)  # Number of matrices in the input
        self.__output_size = output_size  # Number of matrices in the output
        self.__matrix_length = matrix_length
        self.__matrix_size = self.__matrix_length * self.__matrix_length
        self.__pixel_mode = pixel_mode

        if layers_size is not None:
            self.__layers__size = [x*self.__matrix_size for x in layers_size]
        else:
            first_size = self.__input_size*self.__matrix_size
            if self.__pixel_mode:
                second_size = self.__matrix_size
            else:
                second_size = self.__output_size*self.__matrix_size*2
            self.__layers__size = [first_size, second_size]

        self.__layers_size = [8*self.__matrix_size, 8*self.__matrix_size]

        layers = list()
        layers.append(nn.Linear(self.__input_size * self.__matrix_size, self.__layers_size[0]))
        for layer_index in range(len(self.__layers_size)-1):
            layers.append(nn.Linear(self.__layers_size[layer_index], self.__layers_size[layer_index+1]))
        if self.__pixel_mode:
            layers.append(nn.Linear(self.__layers_size[-1], 1))
        else:
            layers.append(nn.Linear(self.__layers_size[-1], self.__output_size*self.__matrix_size))

        self.__layers = nn.ModuleList(layers)



    def forward(self, batch):
        batch = batch[:, self.__input_matrices, :, :].view(-1, self.__input_size * self.__matrix_size)

        for layer_index in range(len(self.__layers)-1):
            batch = F.relu(self.__layers[layer_index](batch))
        batch = torch.tanh(self.__layers[-1](batch))

        if not self.__pixel_mode:
            batch = batch.view(-1, self.__output_size, self.__matrix_length, self.__matrix_length)
        return batch



class cuboidDataset(data.Dataset):
    def __init__(self, matrices_file_path):
        dataset = np.load(matrices_file_path)

        self.__analogies_index = dataset["analogies"]
        self.__sentence_couples_index = dataset["index"]
        self.__matrices = dataset["matrices"]
        self.__matrix_length = self.__matrices.shape[1]
        self.__matrix_size = self.__matrices.shape[1]*self.__matrices.shape[2]
        assert(self.__matrix_size == 10*10)

        self.__length = len(self.__analogies_index)

        self.__analogies_lengths = dataset["lengths"]

        self.__matrices = torch.from_numpy(self.__matrices)
        self.__matrices = self.__matrices.float()
        self.__analogies_lengths = torch.from_numpy(self.__analogies_lengths)
        self.__analogies_lengths = self.__analogies_lengths.float()


    def getmatrixlength(self):
        return self.__matrix_length

    def getmatrixsize(self):
        return self.__matrix_size

    def __len__(self):
        return self.__length

    def __getitem__(self, index):

        matrices_indices = self.__analogies_index[index]
        return self.__matrices[matrices_indices[:6], :, :], \
            self.__matrices[matrices_indices[6:], :, :]


class matricesLoss(torch.nn.modules.loss.MSELoss):

    def __init__(self, input_matrices=(6, 7, 8)):
        super(matricesLoss, self).__init__(None, None, 'mean')
        self.__input_matrices = input_matrices

    def forward(self, input, target):
        return F.mse_loss(input, target[:, self.__input_matrices, :, :], reduction=self.reduction)


class pixelLoss(torch.nn.modules.loss.MSELoss):

    def __init__(self, input_matrix, input_pixel):
        super(pixelLoss, self).__init__(None, None, 'mean')
        self.__input_matrix = input_matrix
        self.__pixel_x = input_pixel[0]
        self.__pixel_y = input_pixel[1]

    def forward(self, input, target):

        return F.mse_loss(input, target[:, self.__input_matrix, self.__pixel_x, self.__pixel_y].view(-1, 1),
                          reduction=self.reduction)

    #def __call__(self, *input, **kwargs):
        #return super(pixelLoss, self).__call__(self, *input, **kwargs)

def train_matrices():
    print("Is cuda available?", torch.cuda.is_available() )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 4}
    max_epochs = 10

    training_set = cuboidDataset("en-fr.matrices.train.npz")
    training_generator = data.DataLoader(training_set, **params)
    test_set = cuboidDataset("en-fr.matrices.test.npz")
    test_generator = data.DataLoader(test_set, **params)


    matrix_length = training_set.getmatrixlength()
    net = simpleDense(matrix_length, input_matrices=(2, 3, 4, 5), pixel_mode=True).to(device)
    print("Net architecture:")
    print(net)
    criterion = pixelLoss(1, (6, 7))


    print_frequency = 2000
    batch_counter = 0

    train_loss = list()
    test_loss = list()

    for epoch in range(max_epochs):
        
        optimizer = optim.Adam(net.parameters(), lr=0.1**(3+epoch/10))
        running_loss = 0.0
        # Training
        epoch_train_loss = list()
        epoch_train_batch = list()
        for i, (local_batch, local_truths) in enumerate(training_generator):


            local_batch, local_truths = local_batch.to(device), local_truths.to(device)

            optimizer.zero_grad()
            outputs = net(local_batch)
            loss = criterion(outputs, local_truths)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_frequency == print_frequency-1:    # print every 2000 mini-batches
                compare = torch.stack((outputs.view(-1), local_truths[:, 1, 6, 7]), dim=0)
                #printMatrix(compare.cpu().detach().numpy())
                #plt.savefig(str(epoch) + str(i) + '.png')
                #plt.clf()
                #plt.close()
                avg_loss = running_loss / print_frequency
                epoch_train_loss.append(avg_loss)
                epoch_train_batch.append(batch_counter)
                batch_counter+=1


                print('[%d, %5d] train loss: %.5f' %
                      (epoch + 1, i + 1, avg_loss))
                running_loss = 0.0

        train_loss.append([epoch_train_batch, epoch_train_loss])

        epoch_test_loss = list()
        epoch_test_batch = list()
        for i, (local_batch, local_truths) in enumerate(test_generator):

            local_batch, local_truths = local_batch.to(device), local_truths.to(device)

            optimizer.zero_grad()
            outputs = net(local_batch, requires_grad = False)
            loss = criterion(outputs, local_truths, requires_grad = False)

            # print statistics
            running_loss += loss.item()
            if i % print_frequency == print_frequency-1:    # print every 2000 mini-batches
                compare = torch.stack((outputs.view(-1), local_truths[:, 1, 6, 7]), dim=0)
                printMatrix(compare.cpu().detach().numpy())
                plt.savefig(str(epoch) + str(i) + '.png')
                plt.clf()
                plt.close()

                avg_loss = running_loss / print_frequency
                epoch_test_loss.append(avg_loss)
                epoch_test_batch.append(batch_counter)
                batch_counter += 1

                print('[%d, %5d] test loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / print_frequency))
                running_loss = 0.0
        test_loss.append([epoch_test_batch, epoch_test_loss])

        for element in train_loss:
            plt.plot([element[0], element[1]], color="red")
        for element in test_loss:
            plt.plot([element[0], element[1]], color="green")
        plt.savefig('epochloss.png')

        plt.clf()
        plt.close()


train_matrices()
