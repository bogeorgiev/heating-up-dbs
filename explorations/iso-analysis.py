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
dim = 3072
alpha = 1.0e1
num_walks = 10000
num_steps = 100 * alpha
step = 0.5e-1 / math.sqrt(alpha)
compute_empirical = True 
device = "cuda"
experiment_id = str(time.time())

normal_distrib = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
normal_1d = Normal(torch.tensor(0.0), torch.tensor(1.0))

msd = .0
msd_th = step * math.sqrt(num_steps * dim)
radius = msd_th
sigma = radius / math.sqrt(dim)

#hyperplane to hit at dist
dist = 0.02 * msd_th #2.0e0 
t = torch.tensor(num_steps * step**2)

#larger ball params
R = 10. * radius
center = torch.zeros(dim).float().to(device)
center[-1] = -R * 0.991 #0.9715

def inside(x):
    center_batch = center.repeat(x.shape[0], 1).to(device)
    return (x - center_batch).norm(dim=1) > R 

def compute_rel_volume(num_samples=10000, uniform_dist=False):
    if uniform_dist:
        error_rate = torch.randn(num_samples, dim + 2).to(device)
        error_rate = radius * error_rate / error_rate.norm(dim=1).unsqueeze(1)
        error_rate = error_rate[:, :-2]
    else:
        error_rate = sigma * torch.randn(num_samples, dim).to(device)

    error_rate = inside(error_rate).sum()
    rel_vol = error_rate.float() / num_samples

    return rel_vol

def compute_med_dist(num_samples=1000, uniform_dist=False):
    if not uniform_dist:
        #compute med dists with normal perturbations

        #pts inside of ball
        error_rate = sigma * torch.randn(num_samples, dim).to(device)
        #print(error_rate.norm(dim=1).mean(), error_rate.norm(dim=1).std())
        pts_in = error_rate[error_rate.norm(dim=1) <= radius]
        center_batch_in = center.repeat(pts_in.shape[0], 1).to(device)
        dists_in = R - (pts_in - center_batch_in).norm(dim=1)
        dists_in[dists_in < 0] = 0 
      
        #pts outside of ball but in upper half-space
        pts_out = error_rate[error_rate.norm(dim=1) > radius]
        pts_out_up = pts_out[pts_out[:, -1] >= 0]
        dists_out_up = (pts_out_up).norm(dim=1) - radius

        #pts outside of ball but in lower half-space
        pts_out_down = pts_out[pts_out[:, -1] < 0]
        norms = (pts_out_down - center).norm(dim=1) * center.norm()
        norms = norms.unsqueeze(1)
        centers = -center.repeat(pts_out_down.shape[0], 1).unsqueeze(2)
        scalar_prods = torch.bmm((pts_out_down / norms - center).unsqueeze(1), centers).squeeze(1).squeeze(1)
        v = torch.zeros(dim).to(device)
        v[0] = radius
        scalar_prod_bound = torch.dot(-center, -center + v) / ((v - center).norm() * center.norm())
        pts_cone = pts_out_down[scalar_prods >= scalar_prod_bound]
        center_batch_cone = center.repeat(pts_cone.shape[0], 1).to(device)
        dists_cone = R - (pts_cone - center_batch_cone).norm(dim=1)
        dists = torch.cat([dists_in, dists_out_up, dists_cone], dim=0) 

    else:
        #compute med dist with uniform samples inside of ball

        error_rate = torch.randn(num_samples, dim + 2).to(device)
        error_rate = radius * error_rate / error_rate.norm(dim=1).unsqueeze(1)
        error_rate = error_rate[:, :-2]
        center_batch = center.repeat(num_samples, 1).to(device)
        dists = R - (error_rate - center_batch).norm(dim=1)
        dists[dists < 0] = 0

    #print(dists.shape)
    return dists.median()

def compute_hit_prob(num_walks=10000):
    hits = 0.
    living = torch.ones(num_walks).to(device).byte()
    samples = torch.zeros(num_walks, dim).to(device)
    for s in range(int(num_steps)):
        samples += step * torch.randn(num_walks, dim).to(device)
        cut_samples = inside(samples)
        hits += (cut_samples * living).sum()
        living *= ~(cut_samples)
        if living.sum() == 0:
            break
    hit_prob = float(hits) / num_walks
    return hit_prob

def compute_isoper_bound(rel_vol):
    iso_bound = 0
    if rel_vol < 0.5:
        iso_bound = -sigma * normal_1d.icdf(rel_vol)
    return iso_bound


def main():
    rel_vol_uni = compute_rel_volume(uniform_dist=True)
    rel_vol_nor = compute_rel_volume(uniform_dist=False)
    hit_prob = compute_hit_prob(num_walks=10000)
    med_dists = compute_med_dist()
    iso_bound = compute_isoper_bound(rel_vol_nor)

    print("Isoperimetric vs Isocapacitory saturation")
    print("BM run for time: ", t)
    print("RMSD (theoretical): ", msd_th)
    print("Relative volume (Normal perturbation) ", rel_vol_nor)
    print("Relative volume (Uniform perturbation) ", rel_vol_uni)
    print("Hitting Probability:", hit_prob)
    print("Median dists: ", med_dists)
    print("Isoperimetric bound:", iso_bound)
    print("Isoperimetric saturation:", med_dists / iso_bound)
    print("Isocapcitory saturation:", hit_prob / rel_vol_uni)
    print("Hitting Prob of hyperplane at dist", dist, ": ", 2 * normal_1d.cdf(-dist / torch.sqrt(t)))

if __name__ == "__main__":
    main()
