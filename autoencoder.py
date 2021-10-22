#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:11:57 2020

@author: lionel

build an autoencoder from scratch using pytorch and peak at the hidden layer to
observe the latent space of the data.
"""



import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


''' define our network'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.enc1 = nn.Linear(784,256)
        self.enc2 = nn.Linear(256,128)
        self.enc3 = nn.Linear(128,2)
        
        self.dec1 = nn.Linear(2,128)
        self.dec2 = nn.Linear(128,256)
        self.dec3 = nn.Linear(256,784)
    
    def latent(self, z):
        z = torch.tanh(self.enc1(z))
        z = torch.tanh(self.enc2(z))
        return self.enc3(z)

    def forward(self, x):
        # print(x.size())
        x = torch.relu(self.enc1(x))
        x = torch.relu(self.enc2(x))
        x = torch.relu(self.enc3(x))
        x = torch.relu(self.dec1(x))
        x = torch.relu(self.dec2(x))
        x = F.dropout(x, training=self.training)
        x = torch.sigmoid(self.dec3(x))
        
        return x
''' setting a random seed for reproducibility'''
np.random.seed(0)


''' load mnist data-set using torchvision '''
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
imshow(torchvision.utils.make_grid(images))

''' Initialize the network'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)


' Select an appropriate loss function, it is the mean squared error in this case'
criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
print_every = 100

train_epoch_loss=[]
test_epoch_loss=[]
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    test_loss = 0.0

    for i, data in enumerate(zip(train_loader,test_loader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0]
        # print(inputs.shape)
        inputs =  inputs.view(-1, np.prod(inputs.shape[1:])).to(device)
        # print(inputs.shape)
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
        tinputs = tinputs.view(-1, np.prod(tinputs.shape[1:])).to(device)
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
    print(epoch)

print('Finished Training')

''' observe reconstructed images'''
for i, data in enumerate(test_loader, 0):
        ''' get data from the test dataset'''
        inputs, labels = data
        
        ''' display samples '''
        imshow(torchvision.utils.make_grid(inputs))
        plt.title('test samples')
        plt.savefig('autoencoder_test_samples.png')
        
        ''' run the test samples through the autoencoder and
        get the reconstructed output'''
        inputs =  inputs.view(-1, np.prod(inputs.shape[1:])).to(device)
        outputs = net(inputs)
        outputs = outputs.view(-1, 1, 28, 28).cpu()
        ''' display the reconstructed images '''
        imshow(torchvision.utils.make_grid(outputs))
        plt.title('test samples reconstructed')
        plt.savefig('autoencoder_test_reconstructed.png')
        break


''' We want to take a peak at the autoencoder latent space: Our expectations is
to observe a modularity of sort such that classes live in a continuous space but
remain separable'''

latest = iter(test_loader)
latent = []
latent_labels =[]
for i, data in enumerate(test_loader,0):
    while len(latent) <= 5000:
        testimages, testlabels = latest.next()
        testimages =  testimages.view(-1, 784).to(device)
        z = net.latent(testimages)
        z = z.cpu().detach().numpy()
        latent.extend(z)
        latent_labels.extend(testlabels.numpy())
    # print(i)

latent = np.array(latent)
plt.figure()
plt.scatter(latent[:,0], latent[:,1], c= latent_labels, cmap=plt.cm.tab10, s=7)
plt.title('Autoencoder Latent Space MNIST')
plt.colorbar()
plt.tight_layout()
plt.savefig('autoencoder_latent_mnist.png')



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
plt.savefig('autoencoder_perf_mnist.png')