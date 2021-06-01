#definition of a classical NN for 0-9 MNIST data recognition

#importing packages
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time


#global variable definition
learning_rate = 0.001
epochs = 15

##################
#DATA LOADING
#training data
n_classes = 10
n_samples = 100
X_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

#keeping only 100 samples
idx = np.stack([np.where(X_train.targets == i)[0][:n_samples] for i in range(10)])
idx = idx.reshape(-1)

X_train.data = X_train.data[idx] #tensor values
X_train.targets = X_train.targets[idx]#tensor labels
print(X_train.targets.max()) #checking the last value


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
n_samples = 100

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx_test = np.stack([np.where(X_test.targets == i)[0][:n_samples] for i in range(10)])
idx_test = idx_test.reshape(-1)

X_test.data = X_test.data[idx_test] #tensor values
X_test.targets = X_test.targets[idx_test]#tensor labels
print(X_test.targets.max())


test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


#########################
#CREATING THE NN
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) #input = gray scale
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d() #deactivating randomly some neurons to avoid overfitting
        self.fc1 = nn.Linear(256, 64) #input dimension: CH(16) x Matrix_dim
        self.fc2 = nn.Linear(64,n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = self.dropout(x)
        x = x.view(1,-1) #reshaping tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    #not using softmax because the loss function is CrossEntropyLoss



#######################
#TRAINING AND TESTING
#importing packages
import Training_Testing as TT

#Training the NN
#we can use any optimizer, learning rate and cost/loss function to train over multiple epochs

model = Net()
params = list(model.parameters())

optimizer = optim.Adam(params, learning_rate)
loss_func = nn.CrossEntropyLoss()

loss_list = []
model.train() #training the module in training mode(specifying the intention to the layers). Used for dropout or batchnorm

#trainign and evaluating time
begin = time.time()
loss_list = (TT.training_loop(epochs, optimizer, model, loss_func, train_loader))
end = time.time()

#plotting the training graph
plt.figure(num=2)
plt.rcParams.update({'font.size': 12})
plt.plot(loss_list)
plt.title('Classic NN Training convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Cross Entropy Loss')

###########################
#TESTING THE NN
n_test_show = 6
fig, axes = plt.subplots(nrows=1, ncols=n_test_show, figsize=(10, 3))

model.eval()
TT.validate(model, test_loader, loss_func, n_test_show, axes)

print('The training took: ', end - begin,'s to complete')

plt.show()

#RESULTS:
#the TRAINING took 51.28 seconds to complete
#Performance on test data:
#Loss: 0.283, Accuracy: 91.9%

