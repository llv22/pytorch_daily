#!/usr/bin/env python
# coding: utf-8

# # Programming Assignment
# 
# This programming assignment is migrated from tensorflow 2.0 exercise, aiming at familiar with pytorch programming API
# 
# Reference:
# * https://nextjournal.com/gkoehler/pytorch-mnist
# * https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# * https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# * https://blog.csdn.net/touristourist/article/details/100535544
# * Transform - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# ## CNN classifier for the MNIST dataset

# #### The MNIST dataset
# 
# In this assignment, you will use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). It consists of a training set of 60,000 handwritten digits with corresponding labels, and a test set of 10,000 images. The images have been normalised and centred. The dataset is frequently used in machine learning research, and has become a standard benchmark for image classification models. 
# 
# - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
# 
# Your goal is to construct a neural network that classifies images of handwritten digits into one of 10 classes.

# In[2]:


import imp
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# #### Load and preprocess the data

# In[3]:


trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


from PIL import Image
from typing import Tuple, Any

class NormalizedMNIST(datasets.MNIST):    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy() / 255., mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        


# In[4]:


MNIST_train = NormalizedMNIST('./data/minst/', train=True, download=True, transform=trans)
MNIST_train.data.shape


# In[5]:


MNIST_train.data.__getitem__(0)


# In[ ]:


MNIST_test = NormalizedMNIST('./data/minst/', train=False, download=True, transform=trans)
MNIST_test.data.shape


# In[ ]:


batch_size_train = 1000
batch_size_test = 1000

# 1, torchvision.transforms.ToTensor(): swap color axis because
# numpy image: H x W x C
# torch image: C x H x W
train_loader = DataLoader(
    NormalizedMNIST('./data/minst/', train=True, download=True, transform=trans),
    batch_size=batch_size_train, 
    shuffle=True
)
test_loader = DataLoader(
    NormalizedMNIST('./data/minst/', train=False, download=True, transform=trans),
    batch_size=batch_size_test, 
    shuffle=True
)


# In[ ]:


class SequentialNetwork(nn.Module):
    def __init__(self):
        super(SequentialNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.LazyConv2d(out_channels=8, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(10),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)


# In[ ]:


import torch.optim as optim

model = SequentialNetwork()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()


# In[ ]:


from torch.nn.functional import one_hot

def sparse_cross_entropy_loss(y_pred, y_true):
    return torch.mean(-torch.log(torch.sum(one_hot(y_true, num_classes=len(y_pred[0])) * y_pred, axis=1)))


# In[ ]:


def train():
    model.train()
    size = len(train_loader.dataset)
    total_batch = len(train_loader)
    for epoch in range(10):
        accuracy = 0
        for batch, (X, y) in enumerate(train_loader):
            # Compute prediction and loss
            pred = model(X)
            loss = sparse_cross_entropy_loss(pred, y)
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Backpropgation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch == total_batch - 1:
                loss, accuracy = loss.item(), accuracy/size
                print(f'Epoch {epoch} - loss: {loss:>7f}, accuracy: {accuracy:>7f}')


# In[ ]:


def test():
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            test_loss += sparse_cross_entropy_loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[ ]:


train()


# In[ ]:


test()


# In[ ]:




