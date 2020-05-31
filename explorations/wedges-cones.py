import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import math
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from labellines import labelLine, labelLines
import numpy as np
import time

"""
    Evaluation of model cases: cones and wedges (i.e. sets between two intersecting hyperplanes).
    Plotting functions illustrating the isocapacitory saturation (tau) in terms of the opening angle and the dimension are provided. 
"""
class Wedge(nn.Module):
    def __init__(self, angle, h, dim, device):
        super(Wedge, self).__init__()
        self.h = torch.tensor(h).to(device)
        self.angle = angle
        self.dim = dim
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
    def __init__(self, angle, h, dim, device):
        super(Cone, self).__init__()
        self.h = torch.tensor(h)
        self.angle = (math.pi - torch.tensor(angle).float()) /2.
        self.dim = dim
        self.cone = nn.Linear(dim, 1, bias=False)
        self.cone.weight.data = torch.ones(1, dim)
        self.cone.weight.data[0, 0] = - (1 / (torch.tan(self.angle)**2))

    def forward(self, x):
        x[:, 0] = x[:, 0] - self.h
        y = x[:, 0].unsqueeze(1)
        x = x**2
        x = self.cone(x)
        return (x < 0) * (y > 0) 

def compute_mesh(model, model_type, mesh_size=7,
            min_angle=0, max_angle=math.pi, dim=10,
            num_vol_samples=1000000, num_walks=50000,
            step = 1.0e-1, num_steps=100):

    cap_arr = []
    vol_arr = []
    mesh = np.linspace(min_angle, max_angle, mesh_size)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    for angle in mesh:
        if model == Wedge:
            h = 0.1
        else:
            h = 0.

        wedge = model(angle=angle, h=h, dim=dim, device=device).to(device)

        #Generate uniform random in the unit ball of dimension dim
        error_rate = torch.randn(num_vol_samples, dim + 2).to(device)
        error_rate = error_rate / error_rate.norm(dim=1).unsqueeze(1)
        error_rate = error_rate[:, :-2]
        error_rate = (wedge(error_rate)).sum()
        vol = error_rate.float() / num_vol_samples
        vol_arr += [vol.item()]

        #Brownian motion sampling and hitting probablities
        hits = 0
        living = torch.ones(num_walks).unsqueeze(1).to(device).byte()
        samples = torch.zeros(num_walks, dim).to(device)

        for s in range(int(num_steps)):
            samples += step * torch.randn(num_walks, dim).to(device)
            out = wedge(samples)
            hits += (out * living).sum()
            living *= ~out
            if living.sum() == 0:
                break
        cap = torch.tensor(float(hits) / num_walks)
        cap_arr += [cap.item()]

    tau = torch.tensor(cap_arr).float() / torch.tensor(vol_arr).float()
    xs_rough = np.linspace(min_angle, max_angle, mesh_size)
    xs = np.linspace(min_angle, max_angle, 100)
    bspline = make_interp_spline(xs_rough, tau.numpy())
    ys = bspline(xs)

    np.save(model_type + "_xs_angle"+str(angle), xs)
    np.save(model_type + "_ys_angle"+str(angle), ys)
    return xs, ys

def plot_tau_stats(dims, model=Wedge):
    if model == Cone:
        model_type = "Cone"
    else:
        model_type = "Wedge"

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Isocapacitory Saturation for a " + model_type)
    ax.set_ylabel("Saturation")
    ax.set_xlabel("Opening angle of " + model_type)

    for dim in dims:
        alpha = 2.0e0
        num_steps = 100 * alpha
        step = 1.0e-1 / math.sqrt(alpha * dim)
        if model == Wedge:
            min_angle = 0.2 * math.pi
        else:
            min_angle = 0.95 * math.pi 
        xs, ys = compute_mesh(model, model_type, min_angle=min_angle, max_angle=math.pi, dim=dim, mesh_size=7, step=step, num_steps=num_steps)

        ax.plot(xs, ys, label=str(dim)) 

    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.show()


def main():
    dims = [10, 20, 30, 40, 50]
    model = Cone #Toggle between Cone and Wedge 
    plot_tau_stats(dims, model=model)

if __name__ == "__main__":
    main()
