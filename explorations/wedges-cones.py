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
dim = 5
alpha = 2.0e0
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
model_type = "cone"

#hyperplane to hit at dist
dist = 0.1 * msd_th #2.0e0 
mesh_size = 40
t = torch.tensor(num_steps * step**2)
print("BM run for time: ", t)
print("RMSD (theoretical): ", msd_th)
print("Hitting Prob (theoretical): ", 2 * normal_1d.cdf(-dist / torch.sqrt(t)))

class Wedge(nn.Module):
    def __init__(self, angle, h, device):
        super(Wedge, self).__init__()
        self.h = torch.tensor(h).to(device)
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
    def __init__(self, angle, h, device):
        super(Cone, self).__init__()
        self.h = torch.tensor(h)
        self.angle = (math.pi - torch.tensor(angle).float()) /2.

        self.cone = nn.Linear(dim, 1, bias=False)
        self.cone.weight.data = torch.ones(1, dim)
        self.cone.weight.data[0, 0] = - (1 / (torch.tan(self.angle)**2))

    def forward(self, x):
        x[:, 0] = x[:, 0] - self.h
        y = x[:, 0].unsqueeze(1)
        x = x**2
        x = self.cone(x)
        return (x < 0) * (y > 0) 

def compute_tau(model, model_type):
    cap_arr = []
    vol_arr = []
    for j in range(mesh_size + 1):
        #h = dist * (1 / mesh_size) #dist * (float(j) / mesh_size)
        h = 0
        print("Distance: ", h)
        angle = float(j) * math.pi / (mesh_size)
        print("Angle: ", angle)
        wedge = model(angle=angle, h=h, device=device).to(device)

        #Generate uniform random in the unit ball of dimension dim
        error_rate = torch.randn(num_walks, dim + 2).to(device)
        error_rate = radius * error_rate / error_rate.norm(dim=1).unsqueeze(1)
        error_rate = error_rate[:, :-2]

        error_rate = (wedge(error_rate)).sum()
        vol = error_rate.float() / num_walks
        print("Vol: ", vol)
        vol_arr += [vol.item()]

        hits = 0
        living = torch.ones(num_walks).unsqueeze(1).to(device)
        samples = torch.zeros(num_walks, dim).to(device)

        for s in range(int(num_steps)):
            samples += step * torch.randn(num_walks, dim).to(device)
            #samples += step * normal_distrib.sample()

            out = wedge(samples)
            hits += (out * living).sum()
             
            living *= ~out
            if living.sum() == 0:
                break

        cap = hits / num_walks
        print("Cap: ", cap)
        cap_arr += [cap.item()]

        writer.add_scalars(model_type + "CapacityVolume" + experiment_id, {"Hitting Prob": cap,
            "Error Rate": vol}, j)
        writer.add_scalar(model_type + "Tau", cap/vol, j)

    np.save(model_type+"Cap", np.array(cap_arr))
    np.save(model_type+"Vol", np.array(vol_arr))
    writer.close()

if __name__ == "__main__":
    model = Cone
    model_type = "Cone"
    compute_tau(model=model, model_type=model_type) 
