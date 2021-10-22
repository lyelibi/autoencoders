#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:11:57 2020

@author: lionel

Proceeding similarly to previous examples found in this repository. We now consider
the implementation using pytorch of a variational autoencoder to perform image
reconstruction on MNIST.
"""



import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


''' define our network '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.enc1 = nn.Linear(784,256)
        self.zmu = nn.Linear(256,2)
        self.zstd = nn.Linear(256,2)
        
        self.dec1 = nn.Linear(2,256)
        self.dec2 = nn.Linear(256,784)
        
    def latent(self, x):
        x = torch.relu(self.enc1(x))
        mu, _ = self.repar(x)
        return mu
    
    def repar(self, x):
        mu = self.zmu(x)
        logvar = self.zstd(x)
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        
        return eps.mul(std).add_(mu), logvar
    

        

    def forward(self, x):

        x = torch.relu(self.enc1(x))
        mu, logvar = self.repar(x)
        
        x = torch.relu(self.dec1(mu))
        x = torch.sigmoid(self.dec2(x))
        
        return x, mu, logvar
    
''''''''''''''''''''''''''''''''

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


criterion = nn.BCELoss( reduction='sum')
# criterion = nn.MSELoss()
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
        inputs =  inputs.view(-1, 784).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, mu, logvar = net.forward(inputs)
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = criterion(outputs, inputs) + kldivergence
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        ''' predict on the test set'''
        tinputs, tlabels = data[1]
        tinputs = tinputs.view(-1, 784).to(device)
        toutputs, tmu, tlogvar = net.forward(tinputs)
        tkldivergence = -0.5 * torch.sum(1 + tlogvar - tmu.pow(2) - tlogvar.exp())
        tloss = criterion(toutputs, tinputs) + tkldivergence
        test_loss += tloss.item()
        
        
        if i % print_every == print_every - 1:
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
        plt.savefig('vae_test_samples.png')
        
        ''' run the test samples through the autoencoder and
        get the reconstructed output'''
        inputs =  inputs.view(-1, 784).to(device)
        outputs, _, _ = net.forward(inputs)
        outputs = outputs.view(-1, 1, 28, 28).cpu()
        ''' display the reconstructed images '''
        imshow(torchvision.utils.make_grid(outputs))
        plt.title('test samples reconstructed')
        plt.savefig('vae_test_reconstructed.png')
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
plt.title('Variational Autoencoder Latent Space MNIST')
plt.colorbar()
plt.tight_layout()
plt.savefig('vae_latent_mnist.png')

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
plt.savefig('vae_perf_mnist.png')