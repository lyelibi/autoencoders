#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:11:57 2020

@author: lionel

Here using Pytorch we build a convolutional autoencoder for which we would 
like to train and test on the MNIST data-set. The MLP autoencoder was only able
to achieve a test loss of 0.045 after training. We expect a Convnet autoencoder
to perform better.

"""



import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


''' define our convnet autoencoder '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        # print(x.shape)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = torch.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))
                
        return x
''''''''''''''''''''''''''    
    
np.random.seed(0)

transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_trainset,
                                          batch_size=50,
                                          shuffle=True,
                                          num_workers=2)

mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset,
                                          batch_size=50,
                                          shuffle=True,
                                          num_workers=2)

'''  display random images '''

def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    try: npimg = img.numpy()
    except: npimg = img.detach().numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
print_every = 100
train_epoch_loss=[]
test_epoch_loss=[]
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    test_loss = 0.0

    for i, data in enumerate(zip(train_loader,test_loader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0]
        inputs =  inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        ''' predict on the test set'''
        tinputs, tlabels = data[1]
        tinputs = tinputs.to(device)
        toutputs = net.forward(tinputs)
        tloss = criterion(toutputs, tinputs)
        test_loss += tloss.item()
        if i % print_every == print_every-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_every))
            train_epoch_loss.append(running_loss / print_every)
            test_epoch_loss.append(test_loss / print_every)
            running_loss = 0.0
            test_loss = 0.0

print('Finished Training')

''' observe reconstructed images'''
for i, data in enumerate(test_loader, 0):
        ''' get data from the test dataset'''
        inputs, labels = data
        
        ''' display samples '''
        imshow(torchvision.utils.make_grid(inputs))
        plt.title('test samples')
        plt.savefig('convauto_test_samples.png')
        
        ''' run the test samples through the autoencoder and
        get the reconstructed output'''
        inputs =  inputs.to(device)
        outputs = net(inputs)
        outputs = outputs.cpu()
        ''' display the reconstructed images '''
        imshow(torchvision.utils.make_grid(outputs))
        plt.title('test samples reconstructed')
        plt.savefig('convauto_test_reconstructed.png')
        break


xaxis = range(len(train_epoch_loss))
plt.figure()
plt.plot(xaxis,train_epoch_loss,c='red', label='train loss')
plt.plot(xaxis,test_epoch_loss,c='blue', label='test loss')
plt.scatter(xaxis,train_epoch_loss,c='red', s=5)
plt.scatter(xaxis,test_epoch_loss,c='blue', s=5)
plt.xlabel('batches')
plt.ylabel('avg loss per %s mini-batches' % print_every)
plt.legend()
plt.tight_layout()
plt.savefig('convauto_perf_mnist.png')