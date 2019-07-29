#!/usr/bin/env python
# coding: utf-8

# # Analyses on the learning ability of the different architectures
# 
# We will analyse the learning ability of the different architectures according to their size (number of learnable parameters).
# In these experiences, matrices have dimension of $10 \times 10$.
# 

# In[ ]:


from math import sqrt

import matplotlib.pyplot as plt
import torch

from neuralNetwork import *

# Choose te best device according to the host machine
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loads datasets
training_set = cuboidDataset("en-fr.matrices.train.npz", device=device)
validation_set = cuboidDataset("en-fr.matrices.validation.npz", device=device)
#test_set = cuboidDataset("en-fr.matrices.test.npz", device=device)

matrix_length = training_set.getmatrixlength()
sizes_to_try = list(range(100000, 1000000, 100000))


# ## Linear model:
# There is no parameters that changes the number of learnable parameters.

# In[ ]:


net = SimpleLinear(matrix_length)
net_size = numberofparameters(net)


# In[ ]:


converged_loss = train(net, 
                       training_set, 
                       validation_set,
                       save_path="./networks/linear_" + str(net_size))

plt.plot([net_size], [converged_loss], label='SimpleLinear')
print("Linear model: Number of parameters: ", net_size, " ; Converged loss: ", converged_loss)


# ## Fully connected model:
# 
# This model give good results when it used with two or three hidden layers.
# We will try these two modes, each hidden layers will have the same number of neurons since
# we don't have any rule for that.
# 
# We will call $n$ the number of neurons on each layers and $S$ the number of learnable parameters.
# 
# ### With two hidden layers:
# 
# We have:
# $S = n^2 + 902n + 300$
# 
# So:
# $n = -\frac{902}{2} + \sqrt{S + (\frac{902}{2})^2 - 300}$

# In[ ]:


def S_from_n(n):
    return n*(n + 902) + 300

def n_from_S(S):
    return sqrt(S + 451**2 - 300) - 451

loss_list = list()
size_list = list()
for S in sizes_to_try:
    n = round(n_from_S(S))
    size_list.append(n)
    net = FullyConnectedNetwork(matrix_length, layers_size=[n, n])
    net_size = numberofparameters(net)
    
    converged_loss = train(net, 
                       training_set, 
                       validation_set,
                           save_path="./networks/fullyconnected2hl_" + str(net_size))

    print("Fully Connected model (2hl): Number of parameters: ", net_size, " ; Converged loss: ", converged_loss)
    loss_list.append(converged_loss)

plt.plot(size_list, loss_list, label='FullyConnected (2hl)')


# ### With three hidden layers:
# 
# We have:
# $S = 2n^2 + 903n + 300$
# 
# So:
# $n = \frac{-903 + \sqrt{8S + 903^2 - 8\times300}}{4}$

# In[ ]:


def S_from_n(n):
    return n*(2*n + 903) + 300

def n_from_S(S):
    return (sqrt((8*S) + (903**2) - (8*300))-903)/4

loss_list = list()
size_list = list()
for S in sizes_to_try:
    n = round(n_from_S(S))
    size_list.append(n)
    net = FullyConnectedNetwork(matrix_length, layers_size=[n, n, n])
    net_size = numberofparameters(net)
    
    converged_loss = train(net, 
                       training_set, 
                       validation_set,
                           save_path="./networks/fullyconnected3hl_" + str(net_size))

    print("Fully Connected model (3hl): Number of parameters: ", net_size, " ; Converged loss: ", converged_loss)
    loss_list.append(converged_loss)

plt.plot(size_list, loss_list, label='FullyConnected (3hl)')


# ## Matrix Channeled model:
# 
# 
# ### With two hidden layers:
# 
# We have:
# $S = 3n^2 + 1506n + 300$
# 
# So:
# $n = \frac{-1506 + \sqrt{12S + 2264436}}{6}$

# In[ ]:


def S_from_n(n):
    return n*(3*n + 1506) + 300

def n_from_S(S):
    return (sqrt(12*S + 2264436)-1506)/6

size_list = list()
for S in sizes_to_try:
    n = round(n_from_S(S))
    size_list.append(n)
    net = FullyConnectedNetwork(matrix_length, layers_size=[n, n])
    net_size = numberofparameters(net)
    converged_loss = train(net, 
                       training_set, 
                       validation_set,
                           save_path="./networks/matrixchanneled2hl_" + str(net_size))

    print("Matrix channeled model (2hl): Number of parameters: ", net_size, " ; Converged loss: ", converged_loss)
    loss_list.append(converged_loss)

plt.plot(size_list, loss_list, label='Matrix_channeled (2hl)')


# ### With three hidden layers:
# 
# We have:
# $S = 6n^2 + 1509n + 300$
# 
# So:
# $n = \frac{-1509 + \sqrt{24S + 2269881}}{12}$

# In[ ]:


def S_from_n(n):
    return n*(6*n + 1509) + 300

def n_from_S(S):
    return (sqrt(34*S + 2269881)-1509)/12

size_list = list()
for S in sizes_to_try:
    n = round(n_from_S(S))
    size_list.append(n)
    net = FullyConnectedNetwork(matrix_length, layers_size=[n, n, n])
    net_size = numberofparameters(net)
    converged_loss = train(net, 
                       training_set, 
                       validation_set,
                           save_path="./networks/matrixchanneled3hl_" + str(net_size))

    print("Matrix channeled model (3hl): Number of parameters: ", net_size, " ; Converged loss: ", converged_loss)
    loss_list.append(converged_loss)

plt.plot(size_list, loss_list, label='Matrix_channeled (3hl)')


# ## Pixel Channeled model:
# 
# 
# ### With two hidden layers:
# 
# We have:
# $S = 300n^2 + 120900n + 300$
# 
# So:
# $n = \frac{-120900 + \sqrt{1200S + 14616450000}}{600}$

# In[ ]:


def S_from_n(n):
    return n*(300*n + 120900) + 300

def n_from_S(S):
    return (sqrt(1200*S + 14616450000)-120900)/600

size_list = list()
for S in sizes_to_try:
    n = round(n_from_S(S))
    size_list.append(n)
    net = FullyConnectedNetwork(matrix_length, layers_size=[n, n])
    net_size = numberofparameters(net)
    converged_loss = train(net, 
                       training_set, 
                       validation_set,
                           save_path="./networks/pixelchanneled2hl_" + str(net_size))
    print("Pixel channeled model (2hl): Number of parameters: ", net_size, " ; Converged loss: ", converged_loss)
    loss_list.append(converged_loss)

plt.plot(size_list, loss_list, label='Pixel_channeled (2hl)')



# ### With three hidden layers:
# 
# We have:
# $S = 600n^2 + 121200n + 300$
# 
# So:
# $n = \frac{-121200 + \sqrt{2400S + 14688720000}}{1200}$

# In[ ]:


def S_from_n(n):
    return n*(600*n + 121200) + 300

def n_from_S(S):
    return (sqrt(2400*S + 14688720000)-121200)/1200

size_list = list()
for S in sizes_to_try:
    n = round(n_from_S(S))
    size_list.append(n)
    net = FullyConnectedNetwork(matrix_length, layers_size=[n, n, n])
    net_size = numberofparameters(net)
    converged_loss = train(net, 
                       training_set, 
                       validation_set,
                           save_path="./networks/pixelchanneled3hl_" + str(net_size))

    print("Pixel channeled model (3hl): Number of parameters: ", net_size, " ; Converged loss: ", converged_loss)
    loss_list.append(converged_loss)

plt.plot(size_list, loss_list, label='Pixel_channeled (3hl)')


# In[ ]:


plt.xlabel("Size of the network (number of learnable parameters).")
plt.ylabel("Average loss after convergence.")
plt.savefig("Architecture_comparisons.pdf", bbox_inches='tight')
plt.show()


# In[ ]:




