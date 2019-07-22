import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
from math import sqrt
from itertools import cycle

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
    def __init__(self, matrix_length, input_matrices=(0, 1, 2, 3, 4, 5), output_size=1):
        super(simpleLinear, self).__init__()
        self.__input_matrices = input_matrices
        self.__matrix_length = matrix_length
        self.__matrix_size = self.__matrix_length * self.__matrix_length
        self.__input_size = len(input_matrices)
        self.__output_size = output_size
        self.__linearLayer = nn.Linear(self.__input_size * self.matrix_size, self.__output_size*self.matrix_size)

    def forward(self, batch):
        batch = batch[:, self.__input_matrices, :, :].view(-1, self.__input_size * self.matrix_size)
        batch = self.__linearLayer(batch).view(-1, self.__matrix_length, self.__matrix_length)
        return batch


class simpleDense(nn.Module):

    def __init__(self, matrix_length, layers_size, input_matrices, output_size=1, pixel_mode=False):
        super(simpleDense, self).__init__()
        self.__input_matrices = input_matrices
        self.__input_size = len(input_matrices)  # Number of matrices in the input
        self.__output_size = output_size  # Number of matrices in the output
        self.__matrix_length = matrix_length
        self.__matrix_size = self.__matrix_length * self.__matrix_length
        self.__pixel_mode = pixel_mode


        self.__layers_size = layers_size



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

    def __init__(self, input_matrices):
        super(matricesLoss, self).__init__(None, None, 'mean')
        self.__input_matrices = [x-6 for x in input_matrices]

    def forward(self, input, target):
        return F.mse_loss(input, target[:, self.__input_matrices, :, :], reduction=self.reduction)


class pixelLoss(torch.nn.modules.loss.MSELoss):

    def __init__(self, input_matrix, input_pixel):
        super(pixelLoss, self).__init__(None, None, 'mean')
        self.__input_matrix = input_matrix - 6
        self.__pixel_x = input_pixel[0]
        self.__pixel_y = input_pixel[1]

    def forward(self, input, target):

        return F.mse_loss(input, target[:, self.__input_matrix, self.__pixel_x, self.__pixel_y].view(-1, 1),
                          reduction=self.reduction)



def smoothing(l, level=3):
    result_list = list()
    for i in range(len(l) - level):
        sum = 0
        for j in range(level):
            sum += l[i+j]
        result_list.append(sum/level)
    return result_list


def train(net, training_set, test_set, truth_index, pixel_mode=False ,save_path=None, verbose=False, initial_lr=0.01, decay=0.5, num_workers=0):
    print("Is cuda available?", torch.cuda.is_available() )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': num_workers}

    net.to(device)


    training_generator = data.DataLoader(training_set, **params)
    test_generator = cycle(data.DataLoader(test_set, **params))

    print("Generators loaded")


    print("Net architecture:")
    print(net)

    if pixel_mode:
        criterion = pixelLoss(truth_index[0], (truth_index[1], truth_index[2]))
    else:
        criterion = matricesLoss(truth_index)

    print_frequency = 1000
    average_test_loss = float("inf")
    continue_training = True
    epoch = 0

    while continue_training:
        
        
    
        optimizer = optim.Adam(net.parameters(), lr=initial_lr*(0.1 ** (decay*epoch)))
        running_test_loss = 0.0
        running_training_loss = 0.0
        # Training
        epoch_train_loss = list()
        epoch_test_loss = list()
        for train_index, (train_batch, train_truth) in enumerate(training_generator):
            batch, truth = train_batch.to(device), train_truth.to(device)

            optimizer.zero_grad()
            outputs = net(batch)
            training_loss = criterion(outputs, truth)
            epoch_train_loss.append(training_loss.item())
            training_loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch, truth = next(test_generator)
                batch, truth = batch.to(device), truth.to(device)

                outputs = net(batch)
                test_loss = criterion(outputs, truth)
                epoch_test_loss.append(test_loss.item())

            if verbose:
                running_test_loss += test_loss.item()
                running_training_loss += training_loss.item()
                if train_index % print_frequency == print_frequency - 1:    # print every print_frequency mini-batches
                    # Attention lookup
                    #optimizer.zero_grad()
                    #outputs[0].backward()
                    #params = [x.grad.cpu().detach().numpy() for x in list(net.parameters())]
                    #product_param =  params[4].dot(params[2]).dot(params[0]).reshape(40, 10)
                    #printMatrix(product_param)
                    #plt.show()

                    # Error lookup (to add to the loss function and refactor)
                    #compare = torch.stack((outputs.view(-1), local_truths[:, 1, 6, 7]), dim=0)
                    #compare = torch.stack((outputs.view(-1), local_truths[:, 1, 1, 1]), dim=0)
                    #printMatrix(compare.cpu().detach().numpy())
                    #plt.savefig("./trainimg/" + str(epoch) + str(i) + '.png')
                    #plt.clf()
                    #plt.close()

                    print('[%d, %5d] training loss: %.5f test loss: %.5f' %
                          (epoch, train_index+1, running_training_loss/print_frequency, running_test_loss/print_frequency))
                    running_training_loss = 0.0
                    running_test_loss = 0.0




        #plt.plot(smoothing(epoch_train_loss, level=1000), color="red", label="train")
        #plt.plot(smoothing(epoch_test_loss, level=1000), color="green", label="test")
        #plt.savefig('epoch' + str(epoch) + '.loss.pdf')
        #plt.clf()
        #plt.close()



        new_average_test_loss = sum(epoch_test_loss[-1000:])/1000
        if new_average_test_loss < average_test_loss:
            average_test_loss = new_average_test_loss
            print("epoch", epoch, " loss: ", average_test_loss)
            if save_path is not None:
                torch.save(net.state_dict(), save_path)
            epoch += 1
        else:
            continue_training = False
    return float(average_test_loss)




training_set = cuboidDataset("en-fr.matrices.train.npz")
test_set = cuboidDataset("en-fr.matrices.test.npz")
matrix_length = training_set.getmatrixlength()

one_layer_loss = np.array((10, 1))
for i in progressbar(range(1, 11)):
    net = simpleDense(matrix_length, input_matrices=(2, 3, 4, 5), pixel_mode=True, layers_size=[i*10])
    final_loss = train(net, training_set, test_set, truth_index=(7, 6, 7), pixel_mode=True)
    one_layer_loss[i, 1] = final_loss
    plt.matshow(one_layer_loss)
    plt.savefig("one_layer_loss.pdf")
    plt.clf()
    plt.close()

index_list = list()
for i in range(1, 11):
    for j in range(1, 11):
        index_list.append(i*j, i, j)
index_list = sorted(index_list)

two_layer_loss = np.array((10, 10))
for _, i, j in progressbar(index_list):
    net = simpleDense(matrix_length, input_matrices=(2, 3, 4, 5), pixel_mode=True, layers_size=[i*10, j*10])
    final_loss = train(net, training_set, test_set, truth_index=(7, 6, 7), pixel_mode=True)
    one_layer_loss[i, j] = final_loss
    plt.matshow(one_layer_loss)
    plt.savefig("two_layer_loss.pdf")
    plt.clf()
    plt.close()





print(final_loss)
