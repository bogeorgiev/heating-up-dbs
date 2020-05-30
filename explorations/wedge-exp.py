import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import math
from matplotlib import pyplot as plt
import numpy as np
import time

writer = SummaryWriter()
dim = 1000
alpha = 1.0e1
num_walks = 10000
num_steps = 100 * alpha
step = 2.0e-1 / math.sqrt(alpha)
compute_empirical = True 
device = "cuda"
experiment_id = str(time.time())

normal_distrib = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
normal_1d = Normal(torch.tensor(0.0), torch.tensor(1.0))

msd = .0
msd_th = step * math.sqrt( num_steps * dim)
radius = msd_th

#hyperplane to hit at dist
dist = 0.1 * msd_th #2.0e0 
mesh_size = 20
t = torch.tensor(num_steps * step**2)
print("BM run for time: ", t)
print("RMSD (theoretical): ", msd_th)
print("Hitting Prob (theoretical): ", 2 * normal_1d.cdf(-dist / torch.sqrt(t)))

class Wedge(nn.Module):
    def __init__(self, angle, h):
        super(Wedge, self).__init__()
        self.h = torch.tensor(h)
        self.angle = angle
        self.normal1 = nn.Linear(dim, 1, bias=True)
        self.normal2 = nn.Linear(dim, 1, bias=True)

        self.normal1.weight.data = torch.zeros(1, dim)
        self.normal1.weight.data[0, 0] = 1.

        self.normal2.weight.data = torch.zeros(1, dim)
        self.normal2.weight.data[0, 0] = torch.cos(torch.tensor(self.angle))
        self.normal2.weight.data[0, 1] = torch.sin(torch.tensor(self.angle))
        self.normal2.weight.data = -self.normal2.weight.data

        self.normal1.bias.data = self.h
        self.normal2.bias.data = -self.h

    def forward(self, x):
        y = self.normal1(x)
        z = self.normal2(x) 
        out = (y > 0) * (z > 0)
        return out 

class Cone(nn.Module):
    def __init__(self):
        super(Wedge, self).__init__()
        self.normal1 = nn.Linear(dim, 1)
        self.normal2 = nn.Linear(dim, 1)
        self.d = 2.

    def forward(self, x):
        x = self.normal1(x)

        return x

def compute_capacity():
    for j in range(mesh_size):

        h = dist * (float(j) / mesh_size)
        normal1 = torch.zeros(dim).float().to(device)
        normal2 = torch.zeros(dim).float().to(device)
        normal1[0] = 1.
        normal2[0] = 1.
        normal2[1] = 0.3
        normal2 = -normal2
        normal2 = normal2 / normal2.norm()

        if (compute_empirical):
            error_rate = torch.randn(num_walks, dim + 2).to(device)
            error_rate = radius * error_rate / error_rate.norm(dim=1).unsqueeze(1)
            error_rate = error_rate[:, :-2]
            error_rate1 = (error_rate * normal1.repeat(num_walks, 1)).sum(dim=1)
            error_rate2 = (error_rate * normal2.repeat(num_walks, 1)).sum(dim=1)

            error_rate1 = (error_rate1 > h)
            error_rate2 = (error_rate2 > -h)
            error_rate = (error_rate1 * error_rate2).sum()
            vol = error_rate.float() / num_walks

        else:
            vol_cap = math.pi * ((radius - h)**2) * (2*radius + h) / 3
            vol_ball = 4 * math.pi * (radius**3) / 3
            vol = float(vol_cap) / vol_ball

        hits = 0
        living = torch.ones(num_walks).to(device)
        samples = torch.zeros(num_walks, dim).to(device)
        if (compute_empirical):
            for s in range(int(num_steps)):
                samples += step * torch.randn(num_walks, dim).to(device)
                #samples += step * normal_distrib.sample()

                samples1 = (samples * normal1.repeat(num_walks, 1)).sum(dim=1)
                samples2 = (samples * normal2.repeat(num_walks, 1)).sum(dim=1)

                samples1 = (samples1 > h)
                samples2 = (samples2 > -h)
                hits += ((samples1 * samples2) * living).sum()
                 
                living *= ~(samples1 * samples2)
                if living.sum() == 0:
                    break

            cap = hits / num_walks

        else:
            cap = 2 * normal_1d.cdf(-h / ( torch.sqrt(t)))
        print("Computed point ", j)
        writer.add_scalars("WedgeExperiments" + experiment_id, {"Hitting Prob": cap,
            "Error Rate": vol, "Tau": cap/vol}, j)

    writer.close()

if __name__ == "__main__":
    wedge = Wedge(angle=math.pi/6, h=2.) 
    x = torch.randn(3, dim)
    print(wedge(x))
    print(wedge.normal1.weight.data)
    wedge.normal1.weight.data = torch.zeros(1, dim)
    wedge.normal1.bias.data = 10 * torch.ones(1)
    print(wedge(x))
    print(wedge.normal1.bias)
        
