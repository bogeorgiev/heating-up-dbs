import torch
import torch.nn as nn

import numpy as np

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x, pert=None):
        if pert == None:
            pert = torch.zeros(len(x), 32, 14, 14).to('cuda:0')
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        #out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


class SubstituteModel(nn.Module):

    def __init__(self):
        super(SubstituteModel, self).__init__()
        self.linear1 = nn.Linear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out


# some simpler models
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)



DNN2 = nn.Sequential(Flatten(), nn.Linear(784, 200), nn.ReLU(),
                         nn.Linear(200, 10))


"""
DNN4 = nn.Sequential(Flatten(), nn.Linear(784, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 100), nn.ReLU(),
                         nn.Linear(100, 10))


"""
"""
CNN = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                        Flatten(),
                        nn.Linear(7*7*64, 100), nn.ReLU(),
                        nn.Linear(100, 10))
"""


class DNN4(nn.Module):
    def __init__(self):
        super(DNN4, self).__init__()
        self.linear1 = nn.Linear(784, 200)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(200, 100)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(100, 100)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(100, 10)

    def forward(self, x, pert=None):
        x = x.view(x.shape[0], -1)
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.relu3(self.linear3(x))
        x = self.linear4(x+pert)
        return x



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.relu4 = nn.ReLU()
        self.linear1 = nn.Linear(7*7*64, 100)
        self.relu5 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)
        

    def forward(self, x, pert=None):
        x = self.relu1(self.conv1(x))
        print('layer12: ', x.size())
        x = self.relu2(self.conv2(x))
        print('layer23: ', x.size())
        x = self.relu3(self.conv3(x))
        print('layer34: ', x.size())
        x = self.relu4(self.conv4(x))
        print('layer45: ', x.size())
        x = x.view(x.shape[0], -1)
        print('layer56: ', x.size())
        x = self.relu5(self.linear1(x))
        print('layer67: ', x.size())
        x = self.linear2(x)
        print('-------------------------')
        return x


