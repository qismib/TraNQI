#!/usr/bin/env python
# coding: utf-8


import numpy as np
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *

# importing packages
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *

import time

# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()



# GLOBAL VARIABLE DEFINITION
n_qubits = 3
simulator = qiskit.Aer.get_backend('qasm_simulator')
n_shots = 1000
shift = np.pi / 2
learning_rate = 0.008
epochs = 10


#chosing the real quantum computer or the simulator
from qiskit.providers.ibmq import least_busy
# We execute on the least busy device (among the actual quantum computers)
#simulator = least_busy(provider.backends(operational = True, simulator=False, status_msg='active',
#                                       filters=lambda x: x.configuration().n_qubits > 1)) 
simulator = provider.get_backend('ibmq_athens')
print("We are executing on...",simulator)
print("It has",simulator.status().pending_jobs,"pending jobs")


import itertools

# creating a list of all possible outputs of a quantum circuit, used for the expectation value
def create_QC_OUTPUTS(n_qubits):
    measurements = list(itertools.product([1, 0], repeat=n_qubits))
    return [''.join([str(bit) for bit in measurement]) for measurement in measurements]




# Hadamard gate (superposition) + Rotation_Y (trainable parameter) + Measure_Z. Returning the frequency of results (00, 01, 10, 11 for 2 qubits)
class QuantumCircuit:
    """
    This class provides an interface to interact with our Quantum Circuit
    """

    def __init__(self, n_qubits, backend, shots):

        # -----Circuit definition
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.parameters = qiskit.circuit.ParameterVector('parameters', n_qubits)

        all_qubits = [i for i in range(n_qubits)]  # qubits vector

        self._circuit.h(all_qubits)  # over all the qubits
        self._circuit.barrier()
        for k in range(n_qubits):
            self._circuit.ry(self.parameters[k], k)
        self._circuit.measure_all()
        # -----

        self.backend = backend
        self.shots = shots
        self.QC_OUTPUTS = create_QC_OUTPUTS(n_qubits)

    def expectation_Z_parameters(self, counts, shots, n_qubits):
        expects = np.zeros(n_qubits)
        i = 0
        for key in counts.keys():
            percentage = counts[key] / shots
            if (i < 3):
                expects[i] += percentage
                i += 1
            else:
                i = 0
        return expects

    def expectation_Z_counts(self, counts, shots, n_qubits):
        expects = np.zeros(len(self.QC_OUTPUTS))

        for k in range(len(self.QC_OUTPUTS)):
            key = self.QC_OUTPUTS[k]
            perc = counts.get(key, 0) / shots
            expects[k] = perc
        return expects

    def run(self, thetas):
        # acting on a simulator
        thetas = thetas.squeeze()
        #        print(thetas)
        p_circuit = self._circuit.bind_parameters({self.parameters[k]: thetas[k].item() for k in range(self.n_qubits)})
        job_sim = qiskit.execute(p_circuit,
                                 self.backend,
                                 shots=self.shots)

        result = job_sim.result()
        counts = result.get_counts(p_circuit)
        #        print('C: ', counts)

        #        expectation_parameters = self.expectation_Z_parameters(counts, self.shots, self.n_qubits)
        expectation_counts = self.expectation_Z_counts(counts, self.shots, self.n_qubits)
        return expectation_counts


circuit = QuantumCircuit(n_qubits, simulator, n_shots)

print(circuit._circuit)
circuit._circuit.draw(output = 'mpl', filename = 'N_qubits.png')#to print as a pyplot figure (new library needed)

rotation = torch.Tensor([np.pi / 4] * n_qubits)
exp = circuit.run(rotation)
print('Expected value for rotation pi/4: {}'.format(exp))

class HybridFunction(Function):
    """Hybrid quantum-classical function definition"""

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """Forward pass computation"""

        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        # context variable (it may take multiple values and return them related to the context). Used to keep track for backpropagation

        expectation_z = ctx.quantum_circuit.run(input)  # evaluating model with trainable parameter
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input,
                              result)  # saves a given tensor for a future call to backward (trainable parameter and the result obtained)
        # input = parametri che passo(3), result = risultati del calcolo(2**n_qubit)

        return result

    @staticmethod
    def backward(ctx, grad_output):  # grad_output os previous gradient

        """Backward computation"""

        input, expectation = ctx.saved_tensors  # evaluated in forward
        input = torch.reshape(input, (-1,))
        gradients = torch.Tensor()

        # iterating to evaluate gradient
        for k in range(len(input)):
            # shifting parameters
            shift_right, shift_left = input.detach().clone(), input.detach().clone()
            shift_right[k] += ctx.shift
            shift_left[k] -= ctx.shift

            # evaluating model after shift
            expectation_right = ctx.quantum_circuit.run(shift_right)
            expectation_left = ctx.quantum_circuit.run(shift_left)

            # evaluating gradient with finite difference formula
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])

            gradients = torch.cat((gradients, gradient.float()))

        result = gradients.float() * grad_output.float()

        return (result).T, None, None


class Hybrid(nn.Module):
    """Hybrid quantum-cassical layer definition"""

    def __init__(self, n_qubits, backend, shots, shift):
        super(Hybrid, self).__init__()

        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift  # parameter shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)  # calling forward and backward




class AddGaussianNoise(object):
    def __init__(self, mean=0., std=5):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


#you can chose whether to use noisy training data or not
# DATA LOADING
# training data
n_samples = 100
mean, std_dv = 0, 0.2
X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))
# keeping only labels 0 and 1
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]  # tensor values
X_train.targets = X_train.targets[idx]  # tensor labels

# making batches (dim = 1). Ir returns an iterable(pytorch tensor)
train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
data_iter = iter(
    train_loader)  # making the iterable an iterator, an object with the next method that can be used in a for cycle

# showing samples
n_samples_show = 6
fig, axes = plt.subplots(nrows=1, ncols=int(n_samples_show), figsize=(10, 3))
# subolot returns the figure and axis that are indipendent as default

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[int(n_samples_show) - 1].imshow(images[0].numpy().squeeze(),
                                         cmap='gray')  # squeeze removes unuseful dim(1). Converting into a numpy vector
    axes[int(n_samples_show) - 1].set_xticks([])
    axes[int(n_samples_show) - 1].set_yticks([])
    axes[int(n_samples_show) - 1].set_title("Labeled: {}".format(targets.item()))

    n_samples_show -= 1



n_samples = 2000

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)




# CREATING THE NN
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # input = gray scale
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()  # deactivating randomly some neurons to avoid overfitting
        self.fc1 = nn.Linear(256, 64)  # input dimension: CH(16) x Matrix_dim (4x4)
        self.fc2 = nn.Linear(64, n_qubits)
        self.hybrid = Hybrid(n_qubits, qiskit.Aer.get_backend('qasm_simulator'), n_shots, shift)
        self.fc3 = nn.Linear(2 ** n_qubits, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout(x)
        x = x.view(1, -1)  # reshaping tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #        print('Params to QC: ', x)
        x = self.hybrid(x)  # calling the forward method for the quantum layer
        #        print('output of QC', x)
        x = torch.Tensor(x.float())
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)  # evaluating probabilities (loss function is a cross entropy)
        return x



def training_loop (n_epochs, optimizer, model, loss_fn, train_loader):
    loss_values = []
    for epoch in range(0, n_epochs, +1):
        total_loss = []

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()#getting rid of previous gradients

            output = model(data)#forward pass
            loss = loss_fn(output, target)
            loss.backward()

            optimizer.step()#updating parameters
            total_loss.append(loss.item())#item transforms into a number

        loss_values.append(sum(total_loss)/len(total_loss))#obtainign the average loss
        print('Training [{:.0f}%]   Loss: {:.4f}'.format(100*(epoch+1)/n_epochs, loss_values[-1]))

    return loss_values



#######################
# TRAINING AND TESTING
# Training the NN
# we can use any optimizer, learning rate and cost/loss function to train over multiple epochs

model = Net()
params = list(model.parameters())

learning_rate = 0.01
optimizer = optim.Adam(params, learning_rate)
loss_func = nn.CrossEntropyLoss()

loss_list = []
model.train()  # training the module in training mode(specifying the intention to the layers). Used for dropout or batchnorm

epochs = 15
begin = time.time()
loss_list = (training_loop(epochs, optimizer, model, loss_func, train_loader))
end = time.time()

# plotting the training graph
plt.figure(num=2)
plt.plot(loss_list)
plt.title('Hybrid NN Training convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Cross Entropy Loss')
plt.savefig('TC_NQ.pdf')



#defining a function to test our net
def validate(model, test_loader, loss_function, n_test, axes):
    correct = 0
    total_loss = []
    count = 0

    with torch.no_grad():  # disabling the gradient as we don't want to update parameters
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)  #evaluating the model on test data

            # evaluating the accuracy of our model
            pred = output.argmax(dim=1,
                                 keepdim=True)  # we are interested in the max value of probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # checking if it matches with label


            #evaluating loss function
            loss = loss_function(output, target)
            total_loss.append(loss.item())

            #printing the resut as images
            if count >= n_test:
                continue
            else:
                axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

                axes[count].set_xticks([])
                axes[count].set_yticks([])
                axes[count].set_title('P: {}'.format(pred.item()))
            count += 1

        print('Performance on test data: \n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'
              .format(sum(total_loss)/len(total_loss),(correct / len(test_loader))*100))



# TESTING THE NN
n_test_show = 6
fig, axes = plt.subplots(nrows=1, ncols=n_test_show, figsize=(10, 3))

model.eval()
validate(model, test_loader, loss_func, n_test_show, axes)

print('The training has taken : ', end - begin, 's to complete')




#testing the NN over set of noisy validation data, with different values of  standard deviation
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=5):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



stop,mean, std_dv= 10, 0, 0.1
for i in range (1, stop):
    
    print('Gaussian noise with std deviation: ', std_dv)

    X_test_n = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor(), AddGaussianNoise(mean, std_dv)]))

    idx = np.append(np.where(X_test_n.targets == 0)[0][:n_samples],
                    np.where(X_test_n.targets == 1)[0][:n_samples])

    X_test_n.data = X_test_n.data[idx]  # tensor values
    X_test_n.targets = X_test_n.targets[idx]  # tensor labels
    test_loader_n = torch.utils.data.DataLoader(X_test_n, batch_size=1, shuffle=True)
    test_iter_n = iter(test_loader_n)

    fig_1, axes_1 = plt.subplots(nrows=1, ncols=n_test_show, figsize=(10, 3))

    model.eval()
    validate(model, test_loader_n, loss_func, n_test_show, axes_1)
    std_dv = std_dv + 0.1



