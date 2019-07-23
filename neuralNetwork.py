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
        self.__linearLayer = nn.Linear(self.__input_size * self.__matrix_size, self.__output_size*self.__matrix_size)

    def forward(self, batch):
        batch = batch[:, self.__input_matrices, :, :].view(-1, self.__input_size * self.__matrix_size)
        batch = self.__linearLayer(batch).view(-1, self.__output_size, self.__matrix_length, self.__matrix_length)
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


class channelDense(nn.Module):

    def __init__(self, matrix_length, layers_size):
        super(channelDense, self).__init__()
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
    def __init__(self, matrices_file_path, device=torch.device("cpu")):
        dataset = np.load(matrices_file_path)

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

    def __init__(self, input_matrices):
        super(matricesLoss, self).__init__(None, None, 'mean')
        self.__input_matrices = [x-6 for x in input_matrices]

    def forward(self, input, target, print_error_path=None):
        if print_error_path is not None:

            input_matrices = input[0].cpu().detach().numpy()
            target_matrices = target[0].cpu().detach().numpy()
            input_matrices = input_matrices.reshape(-1, input_matrices.shape[-1])
            target_matrices = target_matrices.reshape(-1, input_matrices.shape[-1])

            plt.subplot(1, 2, 1)
            plt.matshow(input_matrices, cmap="gray_r", fignum=False)
            plt.subplot(1, 2, 2)
            plt.matshow(target_matrices, cmap="gray_r", fignum=False)
            plt.savefig(print_error_path, bbox_inches='tight')
            plt.clf()
            plt.close()
        return F.mse_loss(input, target[:, self.__input_matrices, :, :], reduction=self.reduction)


class pixelLoss(torch.nn.modules.loss.MSELoss):

    def __init__(self, input_matrix, input_pixel):
        super(pixelLoss, self).__init__(None, None, 'mean')
        self.__input_matrix = input_matrix - 6
        self.__pixel_x = input_pixel[0]
        self.__pixel_y = input_pixel[1]

    def forward(self, input, target, print_error_path=None):
        if print_error_path is not None:
            compare = torch.stack((input.view(-1), target[:, self.__input_matrix, self.__pixel_x, self.__pixel_y]), dim=0)
            printMatrix(compare.cpu().detach().numpy())
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


def train(net, training_set, test_set, truth_index, pixel_mode=False, save_path=None, verbose=False,
          initial_lr=0.001, decay=0.5, num_workers=0, print_error_path=None,
          print_focus_path=None, print_epoch_loss_path=None, patience=10):
    if verbose:
        print("Is cuda available?", torch.cuda.is_available() )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    params = {'batch_size': 64,
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

    print_frequency = 1
    average_test_loss = float("inf")
    continue_training = True
    epoch = 0
    optimizer = optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=decay, patience=patience, verbose=verbose)
    while continue_training:
        
        
    

        running_test_loss = 0.0
        running_training_loss = 0.0
        # Training
        epoch_train_loss = list()
        epoch_test_loss = list()
        for train_index, (train_batch, train_truth) in enumerate(training_generator):
            print_bool = train_index % print_frequency == print_frequency - 1
            #batch, truth = train_batch.to(device), train_truth.to(device)
            batch, truth = train_batch, train_truth

            if print_bool and print_focus_path is not None:
                batch = torch.autograd.Variable(batch, requires_grad=True)
                optimizer.zero_grad()
                outputs = net(batch)
                # Attention lookup
                outputs[0].backward(retain_graph=True)
                params = [x.grad.cpu().detach().numpy() for x in list(net.parameters())[::2][::-1]]
                product_param = np.linalg.multi_dot(params).reshape(-1, matrix_length)
                input_lookup = batch.grad.cpu().detach().numpy()[0][list(net.get_input_index())].reshape(-1, matrix_length)
                printMatrix(input_lookup*product_param)
                plt.savefig(print_focus_path)
                plt.clf()
                plt.close()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                outputs = net(batch)

            training_loss = criterion(outputs, truth)
            epoch_train_loss.append(training_loss.item())
            training_loss.backward()
            optimizer.step()


            with torch.no_grad():
                batch, truth = next(test_generator)
                #batch, truth = batch.to(device), truth.to(device)

                outputs = net(batch)
                if print_bool:
                    # Loss with error printing
                    test_loss = criterion(outputs, truth, print_error_path=print_error_path)
                else:
                    test_loss = criterion(outputs, truth)
                epoch_test_loss.append(test_loss.item())


            running_test_loss += test_loss.item()
            running_training_loss += training_loss.item()
            if print_bool:    # print every print_frequency mini-batches
                avg_test_loss = running_test_loss/print_frequency
                avg_training_loss = running_training_loss/print_frequency
                scheduler.step(avg_test_loss)
                if verbose:
                    print('[%d, %5d] training loss: %.5f test loss: %.5f' %
                          (epoch, train_index+1, avg_training_loss, avg_test_loss))
                running_training_loss = 0.0
                running_test_loss = 0.0



        if print_epoch_loss_path is not None:
            plt.plot(smoothing(epoch_train_loss, level=1000), color="red", label="train")
            plt.plot(smoothing(epoch_test_loss, level=1000), color="green", label="test")
            plt.savefig('epoch' + str(epoch) + "." + print_epoch_loss_path, bbox_inches='tight')
            plt.clf()
            plt.close()



        new_average_test_loss = sum(epoch_test_loss[-1000:])/1000
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





def layers_size_optimize():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_set = cuboidDataset("en-fr.matrices.train.npz", device=device)
    test_set = cuboidDataset("en-fr.matrices.test.npz", device=device)
    matrix_length = training_set.getmatrixlength()

    one_layer_loss = np.ndarray((5, 1))
    for i in progressbar(range(0, 5)):
        net = simpleDense(matrix_length, input_matrices=(2, 3, 4, 5), pixel_mode=True, layers_size=[(i + 1) * 10])
        final_loss = train(net, training_set, test_set, truth_index=(7, 6, 7), pixel_mode=True, verbose=True)
        one_layer_loss[i, 0] = final_loss
        plt.matshow(one_layer_loss)
        plt.colorbar()
        plt.savefig("one_layer_loss.pdf", bbox_inches='tight')
        plt.savefig("one_layer_loss.png", bbox_inches='tight')
        plt.clf()
        plt.close()

    index_list = list()
    for i in range(5):
        for j in range(5):
            index_list.append(((i + 1) * (j + 1), i, j))
    index_list = sorted(index_list)

    two_layer_loss = np.ndarray((5, 5))
    for _, i, j in progressbar(index_list):
        net = simpleDense(matrix_length, input_matrices=(2, 3, 4, 5), pixel_mode=True,
                          layers_size=[(i + 1) * 10, (j + 1) * 10])
        final_loss = train(net, training_set, test_set, truth_index=(7, 6, 7), pixel_mode=True)
        one_layer_loss[i, j] = final_loss
        plt.matshow(two_layer_loss)
        plt.colorbar()
        plt.savefig("two_layer_loss.pdf", bbox_inches='tight')
        plt.savefig("two_layer_loss.png", bbox_inches='tight')
        plt.clf()
        plt.close()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training_set = cuboidDataset("en-fr.matrices.train.npz", device=device)
test_set = cuboidDataset("en-fr.matrices.test.npz", device=device)
matrix_length = training_set.getmatrixlength()
net = channelDense(matrix_length, layers_size=[100, 100])
#final_loss = train(net, training_set, test_set, truth_index=(8, 6, 7), pixel_mode=True, verbose=True, initial_lr=0.001, print_error_path="error.pdf", print_focus_path="focus.pdf")
final_loss = train(net, training_set, test_set, truth_index=(6, 7, 8), verbose=True, initial_lr=0.001, decay=0.5, patience=10, print_error_path="error.pdf")

