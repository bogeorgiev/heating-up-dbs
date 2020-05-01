import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import math
from matplotlib import pyplot as plt
import numpy as np

dim = 3
alpha = 1.0e2
num_walks = 10000
num_steps = 100 * alpha
step = 2.0e-1 / math.sqrt(alpha)
compute_empirical = False

normal_distrib = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
normal_1d = Normal(torch.tensor(0.0), torch.tensor(1.0))

sample = torch.zeros(dim)
msd = .0
msd_th = step * math.sqrt( num_steps * dim)
radius = msd_th

#hyperplane to hit at dist
dist = 0.9 * msd_th #2.0e0 
mesh_size = 200
t = torch.tensor(num_steps * step**2)
print("BM run for time: ", t)
print("RMSD (theoretical): ", msd_th)
print("Hitting Prob (theoretical): ", 2 * normal_1d.cdf(-dist / torch.sqrt(t)))

vols = []
caps = []

for j in range(mesh_size):
    h = dist * (float(j) / mesh_size)

    if (compute_empirical):
        error_rate = torch.randn(num_walks, dim + 2)
        error_rate = radius * error_rate / error_rate.norm(dim=1).unsqueeze(1)
        error_rate = error_rate[:, :-2]
        error_rate = error_rate[:, 0]
        error_rate = (error_rate > h).sum()
        vol = error_rate.float() / num_walks

    else:
        vol_cap = math.pi * ((radius - h)**2) * (2*radius + h) / 3
        vol_ball = 4 * math.pi * (radius**3) / 3
        vol = float(vol_cap) / vol_ball

    vols = vols + [vol]

    hits = 0
    if (compute_empirical):
        for i in range(num_walks):
            walk_has_hit_target = False
            for s in range(int( num_steps)):
                sample += step * torch.randn(dim)
                #sample += step * normal_distrib.sample()

                if sample[0] > h and not walk_has_hit_target:
                    hits += 1
                    walk_has_hit_target = True
                    break

            msd += sample.norm()
            sample = torch.zeros(dim)

        msd = msd / num_walks
        cap = hits / num_walks

    else:
        cap = 2 * normal_1d.cdf(-h / ( torch.sqrt(t)))

    caps = caps + [cap]

tau = torch.tensor(caps) / torch.tensor(vols)
plt.plot([dist * (float(i) / mesh_size) for i in range(mesh_size)], torch.tensor(caps).numpy(), color='r')
plt.plot([dist * (float(i) / mesh_size) for i in range(mesh_size)], torch.tensor(vols).numpy(), color='g')
plt.plot([dist * (float(i) / mesh_size) for i in range(mesh_size)], tau.numpy(), color='b')
plt.show()

if (compute_empirical):
    print("MSD (empirical): ", msd)
    print("Hitting Prob (empirical): ", hits / num_walks)
