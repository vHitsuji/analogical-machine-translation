#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __doc__ string
"""
The purpose of this little script is to compute the needed datasets for the training, the validation and the test
of the neural networks.
"""

__author__ = "Taillandier Valentin"
__copyright__ = "Copyright (C) 2019, Taillandier Valentin"
__license__ = "GPL"
__version__ = "1.0"


import math
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from functools import reduce


def drawMatrix(matrix, x_axis=None, y_axis=None):
    """
    Draw an alignment matrix with given string as axis scale.
    The drawn matrix is stored in the default pyplot container.
    Do not forget to plt.show() if you want to show the matrix
    or to do plt.savefig("mymatrix.pdf") if you want to save the figure.

    This was made for a debug purpose and to make illustration for presentations.

    :param matrix: Numpy 2D-Array, Alignment matrix to show.
    :param x_axis: List of words for the x-axis label.
    :param y_axis: List of words for the y-axis label.
    :return: None
    """
    plt.matshow(matrix, cmap="gray_r")
    if x_axis is not None and y_axis is not None:
        plt.xticks(range(len(y_axis)), y_axis)
        plt.yticks(range(len(x_axis)), x_axis)


class SimpleLinear(nn.Module):
    """
    This architecture is a simple affine regression (only one layer with no activation function).

    """
    def __init__(self, matrix_length, input_matrices=(0, 1, 2, 3, 4, 5), output_size=3):
        """
        Example:
            To make an architecture that takes in input the matrices 0, 1, 2 and 3 and outputs one matrix of size 10x10:
                SimpleLinear(10, input_matrices=(0,1,2,3), output_size=1)
            To make an architecture that takes in input the matrices 0, 1, 2, 3, 4, 5
            and outputs three matrices of size15x15 :
                SimpleLinear(15, input_matrices=(0,1,2,3,4,5), output_size=3)

        :param matrix_length: An integer that represents the lengths of the input matrices
        (of dimension matrix_length x matrix_length).
        :param input_matrices: A tuple of integers which represents the id of the matrices to use when input is given?
        :param output_size: An integer, the number of matrices to output.
        """
        super(SimpleLinear, self).__init__()
        self.__input_matrices = input_matrices
        self.__matrix_length = matrix_length
        self.__matrix_size = self.__matrix_length * self.__matrix_length
        self.__input_size = len(input_matrices)
        self.__output_size = output_size
        self.__linearLayer = nn.Linear(self.__input_size * self.__matrix_size, self.__output_size*self.__matrix_size)

    def forward(self, batch):
        batch = batch[:, self.__input_matrices, :, :].view(-1, self.__input_size * self.__matrix_size)
        batch = self.__linearLayer(batch).view(-1, self.__output_size, self.__matrix_length, self.__matrix_length)
        return batch


class FullyConnectedNetwork(nn.Module):
    """

    """

    def __init__(self, matrix_length, layers_size, input_matrices=(0, 1, 2, 3, 4, 5), output_size=3, pixel_mode=False):
        """
        Example:
        To make an architecture with 2 hidden layers with 10 neurons each
        that takes in input the matrices 0, 1, 2 and 3 and outputs one matrix of size 10x10:
            FullyConnectedNetwork(10, [10, 10], input_matrices=(0,1,2,3), output_size=1)
        To make an architecture with 3 hidden layers with 10, 15 and 20 neurons
        that takes in input the matrices 0, 1, 2, 3, 4, 5
        and outputs three matrices of size15x15:
            FullyConnectedNetwork(15, [10,15,20], input_matrices=(0,1,2,3,4,5), output_size=3)
        To make an architecture with 3 hidden layers with 10, 15 and 20 neurons
        that takes in input the matrices 0, 1, 2, 3 of size 15x15 and
        and outputs one pixel:
            FullyConnectedNetwork(15, [10,15,20], input_matrices=(0,1,2,3), pixel_mode=True)



        :param matrix_length: An integer that represents the lengths of the input matrices
            (of dimension matrix_length x matrix_length).
        :param layers_size: A tuple of integer that represents the number of neurons in each hidden layer.
            The length of the tuple represents the number of hidden layers.
        :param input_matrices: A tuple of integers which represents the id of the matrices to use when input is given?
        :param output_size: An integer, the number of matrices to output. (Not read if pixel_mode=True)
        :param pixel_mode: If true, outputs only one pixel.

        """
        super(FullyConnectedNetwork, self).__init__()
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


    def get_input_index(self):
        return self.__input_matrices

    def forward(self, batch):
        batch = batch[:, self.__input_matrices, :, :].view(-1, self.__input_size * self.__matrix_size)

        for layer_index in range(len(self.__layers)-1):
            batch = F.relu(self.__layers[layer_index](batch))
        batch = torch.tanh(self.__layers[-1](batch))

        if not self.__pixel_mode:
            batch = batch.view(-1, self.__output_size, self.__matrix_length, self.__matrix_length)
        return batch




class multiChannelsLinear(nn.Module):
    """
    Does not work yet.

    This Pytorch's Module class is an attempt to allow more efficient calculation of the application of linear
    modules in parallel.

    In the PixelChanneledNetwork architecture the Gpu has to apply many small transformations and makes the Gpu uses only
    a few percentage of its computing capacity. Reducing these small transformations in a few number of big ones
    allow the Cpu to make the least number of Cuda calls.

    This version doesn't crash and make the computation more efficient in the ChannelNetworkOptimized architecture.
    In my case Gpu usage increases from 20% to 80%.
    By the way, the PixelChanneledNetwork and the PixelChanneledNetworkOptimized architectures should be totally equivalent and
    it seems not to be the case (speed of convergence are different).
    This makes me think that this class is bugged and do not output what is expected.

    """

    __constants__ = ['bias']

    def __init__(self, channels, in_features, out_features, bias=True):
        """
        :param channels: An integer, the number of channels.
        :param in_features: An integer, the number of input features for each channels.
        :param out_features: An integer, the number of output features for each channels.
        :param bias: A boolean, if True, bias will be added (Linear transformations become affine).
        """
        super(multiChannelsLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        self.weight = nn.Parameter(torch.Tensor(channels, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input = input.transpose(0, 2).transpose(0, 1)
        output = self.weight.matmul(input)
        output = output.transpose(0, 1).transpose(0, 2)
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret

    def extra_repr(self):
        return 'channels={}, in_features={}, out_features={}, bias={}'.format(
            self.channels, self.in_features, self.out_features, self.bias is not None
        )


class PixelChanneledNetworkOptimized(nn.Module):
    """
    This is an attempt to optimize the PixelChanneledNetwork class.
    The related neural network architecture remains equivalent but use the Gpu more efficiently.
    Many parallel small transformations are computed in a few number of big ones.
    """

    def __init__(self, matrix_length, layers_size):
        """
        See PixelChanneledNetwork (equivalent architecture).

        :param matrix_length: An integer that represents the lengths of the input matrices
            (of dimension matrix_length x matrix_length).
            This architecture takes directly batches of stack of six matrices,
            i.e. tensors of shape (batch_size, 6, matrix_length, matrix_length).
        :param layers_size: A tuple of integer that represents the number of neurons in each hidden layer.
            The length of the tuple represents the number of hidden layers.
        """
        super(PixelChanneledNetworkOptimized, self).__init__()
        self.__matrix_length = matrix_length
        self.__matrix_size = self.__matrix_length * self.__matrix_length
        self.__layers_size = layers_size

        self.__first_layer0 = nn.Linear(4 * self.__matrix_size, self.__layers_size[0]*self.__matrix_size)
        self.__first_layer1 = nn.Linear(4 * self.__matrix_size, self.__layers_size[0]*self.__matrix_size)
        self.__first_layer2 = nn.Linear(4 * self.__matrix_size, self.__layers_size[0]*self.__matrix_size)

        layers_list = list()
        for layer_index in range(len(self.__layers_size)-1):
            layers_list.append(multiChannelsLinear(3 * self.__matrix_size,
                                                   self.__layers_size[layer_index], self.__layers_size[layer_index+1]))

        layers_list.append(multiChannelsLinear(3 * self.__matrix_size, self.__layers_size[-1], 1))
        self.__layers_list = nn.ModuleList(layers_list)





    def forward(self, batch):

        first_output0 = self.__first_layer0(batch[:, [0, 1, 2, 3], :, :].reshape(-1, 4*self.__matrix_size))\
            .reshape(-1, self.__matrix_size, self.__layers_size[0])
        first_output1 = self.__first_layer1(batch[:, [2, 3, 4, 5], :, :].reshape(-1, 4*self.__matrix_size))\
            .reshape(-1, self.__matrix_size, self.__layers_size[0])
        first_output2 = self.__first_layer2(batch[:, [0, 1, 4, 5], :, :].reshape(-1, 4*self.__matrix_size))\
            .reshape(-1, self.__matrix_size, self.__layers_size[0])
        output = F.relu(torch.cat([first_output0, first_output1, first_output2], dim=1))

        for layer_index in range(len(self.__layers_list) - 1):
            output = F.relu(self.__layers_list[layer_index](output))
        output = torch.tanh(self.__layers_list[-1](output))

        return output.reshape(-1, 3, self.__matrix_length, self.__matrix_length)




class MatrixChanneledNetwork(nn.Module):
    """
    In this architecture, each of the three output matrices will be computed using four matrices (four to one model).
    Each matrix is computed independently using the information of the four known matrices that constraint this matrix.
    Each channel is a sub fully connected network and is equivalent to the FullyConnectedNetwork in its matrix mode.
    This architecture simply embeds all of these sub networks into a big one that can be trained at once.
    """

    def __init__(self, matrix_length, layers_size):
        """
        :param matrix_length: An integer that represents the lengths of the input matrices
        (of dimension matrix_length x matrix_length).
        This architecture takes directly batches of stack of six matrices,
        i.e. tensors of shape (batch_size, 6, matrix_length, matrix_length).
        :param layers_size: A tuple of integer that represents the number of neurons in each hidden layer.
        The length of the tuple represents the number of hidden layers.
        """
        super(MatrixChanneledNetwork, self).__init__()
        self.__matrix_length = matrix_length
        self.__matrix_size = self.__matrix_length * self.__matrix_length
        self.__layers_size = layers_size

        channels_list = list()
        for out_matrix_i in range(3):
            layers_list = list()
            #  Making layers for this matrix
            layers_list.append(nn.Linear(4 * self.__matrix_size, self.__layers_size[0]))
            for layer_index in range(len(self.__layers_size)-1):
                layers_list.append(nn.Linear(self.__layers_size[layer_index], self.__layers_size[layer_index+1]))
            layers_list.append(nn.Linear(self.__layers_size[-1], self.__matrix_size))

            channels_list.append(nn.ModuleList(layers_list))
        self.__channels_list = nn.ModuleList(channels_list)




    def __forward_channel(self, batch, channel_id):

        batch = batch.view(-1, 4 * self.__matrix_size) #  Flatten all matrices (first dim is batch dim)
        layers = self.__channels_list[channel_id]

        for layer_index in range(len(layers) - 1):
            batch = F.relu(layers[layer_index](batch))
        batch = torch.tanh(layers[-1](batch))
        batch = batch.view(-1, self.__matrix_length, self.__matrix_length)
        return batch

    def forward(self, batch):

        matrix_6 = self.__forward_channel(batch[:, [0, 1, 2, 3], :, :], 0)
        matrix_7 = self.__forward_channel(batch[:, [2, 3, 4, 5], :, :], 1)
        matrix_8 = self.__forward_channel(batch[:, [0, 1, 4, 5], :, :], 2)

        return torch.stack([matrix_6, matrix_7, matrix_8], dim=1)



class PixelChanneledNetwork(nn.Module):
    """
    In this architecture, each pixel of each output matrix will be computed using its proper channel,
    each pixel is computed independently using the information of the four known matrices that constraint this pixel.
    Each channel is a sub fully connected network and is equivalent to the FullyConnectedNetwork in its pixel mode.
    This architecture simply embeds all of these sub networks into a big one that can be trained at once.

    This architecture suffer from important lack of efficiency due to too many small calls to the Gpu.
    An attempt to optimize that has been made in the PixelChanneledNetworkOptimized class.

    """

    def __init__(self, matrix_length, layers_size):
        """
        :param matrix_length: An integer that represents the lengths of the input matrices
        (of dimension matrix_length x matrix_length).
        This architecture takes directly batches of stack of six matrices,
        i.e. tensors of shape (batch_size, 6, matrix_length, matrix_length).
        :param layers_size: A tuple of integer that represents the number of neurons in each hidden layer.
        The length of the tuple represents the number of hidden layers.
        """
        super(PixelChanneledNetwork, self).__init__()
        self.__matrix_length = matrix_length
        self.__matrix_size = self.__matrix_length * self.__matrix_length
        self.__layers_size = layers_size

        channels_list = list()
        for out_matrix_i in range(3):
            x_list = list()
            for out_x_i in range(self.__matrix_length):
                y_list = list()
                for out_y_i in range(self.__matrix_length):

                    layers_list = list()

                    #  Making layers for this pixel
                    layers_list.append(nn.Linear(4 * self.__matrix_size, self.__layers_size[0]))
                    for layer_index in range(len(self.__layers_size)-1):
                        layers_list.append(nn.Linear(self.__layers_size[layer_index], self.__layers_size[layer_index+1]))
                    layers_list.append(nn.Linear(self.__layers_size[-1], 1))

                    y_list.append(nn.ModuleList(layers_list))
                x_list.append(nn.ModuleList(y_list))
            channels_list.append(nn.ModuleList(x_list))
        self.__channels_list = nn.ModuleList(channels_list)




    def __forward_channel(self, batch, channel_id):

        batch = batch.view(-1, 4 * self.__matrix_size) #  Flatten all matrices (first dim is batch dim)
        channel = self.__channels_list[channel_id]

        output_matrix = list()
        for x, x_list in enumerate(channel):
            row_list = list()
            for y, layers in enumerate(x_list):
                pixel_list = list()
                output = batch
                for layer_index in range(len(layers) - 1):
                    output = F.relu(layers[layer_index](output))
                output = torch.tanh(layers[-1](output))
                pixel_list.append(output)
                row_list.append(torch.cat(pixel_list, dim=1))
            output_matrix.append(torch.cat(row_list, dim=1))
        output_matrix = torch.stack(output_matrix, dim=1)
        return output_matrix





    def forward(self, batch):

        matrix_6 = self.__forward_channel(batch[:, [0, 1, 2, 3], :, :], 0)
        matrix_7 = self.__forward_channel(batch[:, [2, 3, 4, 5], :, :], 1)
        matrix_8 = self.__forward_channel(batch[:, [0, 1, 4, 5], :, :], 2)

        return torch.stack([matrix_6, matrix_7, matrix_8], dim=1)




class cuboidDataset(data.Dataset):
    """
    Loads a cuboids dataset that can be used in a Pytorch's Dataloader.
    """

    def __init__(self, dataset_file_path, device=torch.device("cpu")):
        """
        :param dataset_file_path: Path to the dataset.
        :param device: Pytorch's device object where to store the dataset.
        """

        dataset = np.load(dataset_file_path)

        self.__analogies_index = dataset["analogies"]
        self.__sentence_couples_index = dataset["index"]
        self.__matrices = dataset["matrices"]
        self.__matrix_length = self.__matrices.shape[1]
        self.__matrix_size = self.__matrices.shape[1]*self.__matrices.shape[2]
        assert(self.__matrix_size == 10*10)

        self.__length = len(self.__analogies_index)


        self.__matrices = torch.from_numpy(self.__matrices).to(device)
        self.__matrices = self.__matrices.float()


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
    """
    Loss criterion builder for matrices using mean square errors.
    """

    def __init__(self, input_matrices):
        """
        :param input_matrices: Index of the matrices to compare.
        """
        super(matricesLoss, self).__init__(None, None, 'mean')
        self.__input_matrices = [x-6 for x in input_matrices]

    def forward(self, input, target, print_error_path=None,):
        if print_error_path is not None:

            input_matrices = input[0].cpu().detach().numpy()
            target_matrices = target[0].cpu().detach().numpy()
            input_matrices = input_matrices.reshape(-1, input_matrices.shape[-1])
            target_matrices = target_matrices.reshape(-1, input_matrices.shape[-1])

            plt.subplot(1, 2, 1)
            plt.matshow(input_matrices, cmap="gray_r", fignum=False)
            plt.subplot(1, 2, 2)
            plt.matshow(target_matrices, cmap="gray_r", fignum=False)
            if print_error_path is not None:
                plt.savefig(print_error_path, bbox_inches='tight')
            plt.clf()
            plt.close()
        return F.mse_loss(input, target[:, self.__input_matrices, :, :], reduction=self.reduction)


class pixelLoss(torch.nn.modules.loss.MSELoss):
    """
    Loss criterion builder for pixel using mean square errors.
    """

    def __init__(self, input_matrix, input_pixel):
        """
        :param input_matrix: The id of the matrix containing the pixel.
        :param input_pixel: A pair of integers giving the position of the pixel to compare.
        """
        super(pixelLoss, self).__init__(None, None, 'mean')
        self.__input_matrix = input_matrix - 6
        self.__pixel_x = input_pixel[0]
        self.__pixel_y = input_pixel[1]

    def forward(self, input, target, print_error_path=None):
        if print_error_path is not None:
            compare = torch.stack((input.view(-1), target[:, self.__input_matrix, self.__pixel_x, self.__pixel_y]), dim=0)
            drawMatrix(compare.cpu().detach().numpy())
            plt.savefig(print_error_path, bbox_inches='tight')
            plt.clf()
            plt.close()
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


def train(net, training_set, test_set, truth_index=(6, 7, 8), pixel_mode=False, save_path=None, verbose=False,
          batch_size=64, initial_lr=0.001, decay=0.9, num_workers=0, average_frequency=30, print_error_path=None,
          print_epoch_loss_path=None, patience=10):

    """
    Train a given neural network.

    :param net: Neural network to train.
    :param training_set: Training dataset.
    :param test_set: Validation dataset.
    :param truth_index:
        If matrix mode (i.e. pixel_mode=False): Index of the matrices in the ground truth to compare with the outputs.
        If pixel mode: Pair of index of the matrix containing the pixel in the ground truth and location of the pixel.
            Example: (1, (4, 5)) to compare with the pixel at x=4, y=5 in the matrix with id 1.
    :param pixel_mode: A boolean, if true activates the pixel mode.
    :param save_path: The path where to save the parameters of the neural network after training.
        Does not store anything by default.
    :param verbose: If true, information about running loss and learning rates will be printed in stdout.
    :param batch_size: Integer, the size of batchs.
    :param initial_lr: The initial learning rate.
    :param decay: When the initial rate has to be decreased, apply new_learning_rate = decay * old_learning_rate
    :param num_workers: Number of workers for loading the batches (If you get errors, put at 0).
    :param average_frequency: The frequency of batchs to average the last losses.
        These last losses are used to draw figures, to feed the learning rate scheduler,
        and decide if a new epoch is needed.
    :param print_error_path: Output path where to store the last output/truth error comparison.
    :param print_epoch_loss_path: Output path where to store the figure of losses per epochs.
    :param patience: Patience for the learning rate scheduler.
        Each average_frequency, an average on validation loss will be given to the scheduler.
        If these average losses have not decreased after patience number calls,
        the scheduler decreases the learning rate.
    :return: The final training loss that has been obtained after converging.
    """

    if verbose:
        print("Is cuda available?", torch.cuda.is_available() )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers}

    net = net.to(device)

    training_generator = data.DataLoader(training_set, **params)
    test_generator = cycle(data.DataLoader(test_set, **params))

    if verbose:
        print("Generators loaded")
        print("Net architecture:")
        print(net)

    if pixel_mode:
        criterion = pixelLoss(truth_index[0], (truth_index[1], truth_index[2]))
    else:
        criterion = matricesLoss(truth_index)

    average_test_loss = float("inf")
    continue_training = True
    epoch = 0
    optimizer = optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=decay, patience=patience, verbose=verbose)

    epoch_train_loss = list()
    epoch_test_loss = list()
    while continue_training:
        
        
    
        epoch_train_loss.append([])
        epoch_test_loss.append([])
        running_test_loss = 0.0
        running_training_loss = 0.0
        # Training
        for train_index, (train_batch, train_truth) in enumerate(training_generator):
            time_to_average = train_index % average_frequency == average_frequency - 1
            batch, truth = train_batch, train_truth

            optimizer.zero_grad()
            outputs = net(batch)

            training_loss = criterion(outputs, truth)
            training_loss.backward()
            optimizer.step()


            with torch.no_grad():
                batch, truth = next(test_generator)

                outputs = net(batch)
                if time_to_average:
                    # Loss with error printing
                    test_loss = criterion(outputs, truth, print_error_path=print_error_path)
                else:
                    test_loss = criterion(outputs, truth)


            running_test_loss += test_loss.item()
            running_training_loss += training_loss.item()
            if time_to_average:    # print every average_frequency mini-batches
                avg_test_loss = running_test_loss/average_frequency
                avg_training_loss = running_training_loss/average_frequency

                epoch_train_loss[-1].append(avg_training_loss)
                epoch_test_loss[-1].append(avg_test_loss)

                scheduler.step(avg_test_loss)
                if verbose:
                    print('[%d, %5d] training loss: %.5f test loss: %.5f' %
                          (epoch, train_index+1, avg_training_loss, avg_test_loss))
                running_training_loss = 0.0
                running_test_loss = 0.0



        if print_epoch_loss_path is not None:
            train_loss_list = list()
            test_loss_list = list()
            epoch_length = [0]
            for past_epoch in range(len(epoch_train_loss)):
                train_loss_list.extend(epoch_train_loss[past_epoch])
                test_loss_list.extend(epoch_test_loss[past_epoch])
                epoch_length.append(epoch_length[-1] + len(epoch_train_loss[past_epoch]))

            plt.plot(train_loss_list, color="red", label="train")
            plt.plot(test_loss_list, color="green", label="test")
            plt.xticks(epoch_length, list(range(len(epoch_length) + 1)))
            plt.savefig('epoch' + str(epoch) + "." + print_epoch_loss_path, bbox_inches='tight')
            plt.clf()
            plt.close()



        new_average_test_loss = epoch_test_loss[-1][-1]
        if new_average_test_loss < average_test_loss*0.9:
            average_test_loss = new_average_test_loss
            if verbose:
                print("epoch", epoch, " loss: ", average_test_loss)
            if save_path is not None:
                torch.save(net.state_dict(), save_path)
            epoch += 1
        else:
            continue_training = False
    return float(average_test_loss)



def numberofparameters(net):
    """
    Returns the number of trainable parameters of a given neural network.
    :param net: The neural network object.
    :return: An integer.
    """
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])





"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training_set = cuboidDataset("en-fr.matrices.train.npz", device=device)
test_set = cuboidDataset("en-fr.matrices.test.npz", device=device)
matrix_length = training_set.getmatrixlength()

net = PixelChanneledNetworkOptimized(matrix_length, layers_size=[100, 100])
print("Number of parameters", numberofparameters(net))
final_loss = train(net, training_set, test_set, truth_index=(6, 7, 8), verbose=True, initial_lr=0.001, decay=0.9, patience=10, print_error_path="error.pdf")

"""