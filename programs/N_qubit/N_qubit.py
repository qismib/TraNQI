#trying to implement the same data recognition but with 3 qubit and the same trainable parameter


#importing packages
import numpy as np
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


#GLOBAL VARIABLE DEFINITION
n_qubits = 3
simulator = qiskit.Aer.get_backend('qasm_simulator')
n_shots = 1000
shift = np.pi/2
learning_rate = 0.01
epochs = 10

#######################
#importing the circuit: importing the definition of the quantum circuit and the extent of the backward and forward method
import QC_N_qubits as QC
circuit = QC.QuantumCircuit(n_qubits, simulator, n_shots)

print(circuit._circuit)
#circuit._circuit.draw(output='mpl')#to print as a pyplot figure (new library needed)

rotation = torch.Tensor([np.pi/4]*n_qubits)
exp = circuit.run(rotation)
print('Expected value for rotation pi/4: {}'.format(exp))


#######################
class Hybrid(nn.Module):
    """Hybrid quantum-cassical layer definition"""

    def __init__(self, n_qubits ,backend, shots, shift):
        super(Hybrid, self).__init__()

        self.quantum_circuit = QC.QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift #parameter shift

    def forward(self, input):
        return QC.HybridFunction.apply(input, self.quantum_circuit, self.shift) #calling forward and backward
'''
x = torch.tensor([np.pi/4]*n_qubits, requires_grad=True)
qc = QC.HybridFunction.apply
y1 = qc(x, circuit, shift)
print('y1 after quantum layer: ', y1.float())
y1 = nn.Linear(2**n_qubits, 1)(y1.float())
y1.backward()
print('x.grad = ',x.grad )

#testing the Quantum circuit gradients descent
qc = QC.HybridFunction.apply

def cost (x):
    target = -1
    #print('x', x)
    expval = qc(x, circuit, shift)[0]
    #print(expval)
    #averaging over all outputs of quantum layer
    val = sum([(i+1)*expval[i] for i in range (2**n_qubits)])/2**n_qubits
    val = torch.abs(val-target)**2
    return val, expval

x = torch.tensor([-np.pi/4]*n_qubits, requires_grad=True)#inizilaing parameters
opt = torch.optim.Adam([x], lr=0.1)
num_epoch = 100
loss_list = []

for i in range(num_epoch):
    opt.zero_grad()
    loss, expval = cost(x)
    #print(loss, expval)
    loss.backward()
    opt.step()
    loss_list.append(loss.item())


plt.figure(num =3)
plt.plot(loss_list)
'''

##################3
#DATA LOADING
#training data
n_samples = 100
X_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#keeping only labels 0 and 1
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples])


X_train.data = X_train.data[idx] #tensor values
X_train.targets = X_train.targets[idx]#tensor labels

#making batches (dim = 1). Ir returns an iterable(pytorch tensor)
train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
data_iter = iter(train_loader)#making the iterable an iterator, an object with the next method that can be used in a for cycle


#showing samples
n_samples_show = 6
fig, axes = plt.subplots(nrows=1, ncols=int(n_samples_show), figsize=(10, 3))
#subolot returns the figure and axis that are indipendent as default

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[int(n_samples_show) - 1].imshow(images[0].numpy().squeeze(), cmap='gray')#squeeze removes unuseful dim(1). Converting into a numpy vector
    axes[int(n_samples_show) - 1].set_xticks([])
    axes[int(n_samples_show) - 1].set_yticks([])
    axes[int(n_samples_show) - 1].set_title("Labeled: {}".format(targets.item()))

    n_samples_show -= 1


#validation data
n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

#########################
#CREATING THE NN
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) #input = gray scale
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d() #deactivating randomly some neurons to avoid overfitting
        self.fc1 = nn.Linear(256, 64) #input dimension: CH(16) x Matrix_dim (4x4)
        self.fc2 = nn.Linear(64,n_qubits)
        self.hybrid = Hybrid(n_qubits, qiskit.Aer.get_backend('qasm_simulator'), n_shots, shift)
        self.fc3 = nn.Linear(2**n_qubits, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = self.dropout(x)
        x = x.view(1,-1) #reshaping tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
#        print('Params to QC: ', x)
        x = self.hybrid(x) #calling the forward method for the quantum layer
#        print('output of QC', x)
        x = torch.Tensor(x.float())
        x = self.fc3(x)
        x = torch.softmax(x, dim = 1) #evaluating probabilities (loss function is a cross entropy)
        return x




#######################
#TRAINING AND TESTING
import Training_Testing as TT

#Training the NN
#we can use any optimizer, learning rate and cost/loss function to train over multiple epochs

model = Net()
params = list(model.parameters())

optimizer = optim.Adam(params, learning_rate)
loss_func = nn.CrossEntropyLoss()

loss_list = []
model.train() #training the module in training mode(specifying the intention to the layers). Used for dropout or batchnorm

begin = time.time()
loss_list = (TT.training_loop(epochs, optimizer, model, loss_func, train_loader))
end = time.time()


#plotting the training graph
plt.figure(num=2)
plt.plot(loss_list)
plt.title('Hybrid NN Training convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Cross Entropy Loss')

###########################
#TESTING THE NN
n_test_show = 6
fig, axes = plt.subplots(nrows=1, ncols=n_test_show, figsize=(10, 3))

model.eval()
TT.validate(model, test_loader, loss_func, n_test_show, axes)

print('The training has taken : ', end - begin , 's to complete')

plt.show()