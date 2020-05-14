import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MLPModel(nn.Module):
    def __init__(self, hidden_nodes=40, input_size=2, num_classes=2):
        super(MLPModel, self).__init__()
        self.hidden_nodes = hidden_nodes
        self.input_size = input_size
        self.num_classes = num_classes

        self.l1 = nn.Linear(self.input_size, self.hidden_nodes, bias=True)
        self.l2 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=True)
        self.l3 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=True)
        self.l4 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=True)
        self.l5 = nn.Linear(self.hidden_nodes, self.num_classes, bias=False)

        self.batch_norm_1 = nn.BatchNorm1d(self.hidden_nodes)
        self.batch_norm_2 = nn.BatchNorm1d(self.hidden_nodes)
        self.batch_norm_3 = nn.BatchNorm1d(self.hidden_nodes)
        self.batch_norm_4 = nn.BatchNorm1d(self.hidden_nodes)
        self.batch_norm_5 = nn.BatchNorm1d(self.hidden_nodes)
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.mlp = nn.Sequential(self.l1, self.batch_norm_1, self.relu,
                                self.l2, self.batch_norm_2, self.relu,
                                self.l3, self.batch_norm_3, self.relu,
                                self.l4, self.batch_norm_4, self.relu,
                                self.l5)

    def forward(self, x):
        x = self.mlp(x)
        return x


