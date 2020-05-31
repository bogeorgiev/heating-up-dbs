import sys
sys.path.append("../")
import torch
from torch.distributions.normal import Normal
import math
import numpy as np


#Basic Brownian Motion Simulation with statistics of root mean squared displacement
def compute_rmsd(compute_empirical_rmsd=False, dim=3072, num_walks=1000, step=0.1, num_steps=100):
    t = torch.tensor(num_steps * step**2)
    normal_1d = Normal(torch.tensor(0.0), torch.tensor(1.0))
    rmsd = torch.tensor(0.0)

    if (compute_empirical_rmsd):
        sample = torch.zeros(dim)
        for i in range(num_walks):
            for s in range(int( num_steps)):
                sample += step * torch.randn(dim)

            rmsd += sample.norm()
            sample = torch.zeros(dim)

        rmsd = rmsd / num_walks

    else:
        rmsd = torch.sqrt(dim * t)

    return rmsd

def main():
    alpha = 1.0e0
    num_steps = 100 * alpha
    step = 0.1 / math.sqrt(alpha)
    emp_rmsd = compute_rmsd(compute_empirical_rmsd=False, step=step, num_steps=num_steps)
    th_rmsd = compute_rmsd(compute_empirical_rmsd=True)
    print("Theoretical RMSD: ", emp_rmsd)
    print("Empirical RMSD: ", th_rmsd)
    print("% Error: ", torch.abs(emp_rmsd - th_rmsd) / th_rmsd)

if __name__=="__main__":
    main()
