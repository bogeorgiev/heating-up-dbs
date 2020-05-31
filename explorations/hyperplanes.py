import sys
sys.path.append("../")
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import math
from matplotlib import pyplot as plt
from labellines import labelLine, labelLines
import numpy as np
from scipy.interpolate import make_interp_spline
from utils.utils import planar_cap


def compute_vol(h, compute_empirical_vol, num_samples=500000, dim=3072):
    if (compute_empirical_vol):
        error_rate = torch.randn(num_samples, dim + 2)
        error_rate = error_rate / error_rate.norm(dim=1).unsqueeze(1)
        error_rate = error_rate[:, :-2]
        error_rate = error_rate[:, 0]
        error_rate = (error_rate > h).sum()
        vol = error_rate.float() / num_samples

    else:
        vol_cap = math.pi * ((radius - h)**2) * (2*radius + h) / 3
        vol_ball = 4 * math.pi * (radius**3) / 3
        vol = float(vol_cap) / vol_ball
    return vol


def compute_cap(h, compute_empirical_cap, dim=3072, num_walks=1000):
    t = torch.tensor(1.0 / dim)
    normal_1d = Normal(torch.tensor(0.0), torch.tensor(1.0))
    hits = 0
    if (compute_empirical_cap):
        for i in range(num_walks):
            walk_has_hit_target = False
            for s in range(int( num_steps)):
                sample += step * torch.randn(dim)
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

    return cap

def compute_mesh(compute_empirical_vol, compute_empirical_cap, dim=3072, dist=0.13, mesh_size=4):
    vols = []
    caps = []
    
    mesh = np.linspace(0, dist, mesh_size)
    for h in mesh:
        vol = compute_vol(h, compute_empirical_vol=compute_empirical_vol, dim=dim)
        vols = vols + [vol]

        cap = compute_cap(h, compute_empirical_cap=compute_empirical_cap, dim=dim)
        caps = caps + [cap]

    tau = torch.tensor(caps) / torch.tensor(vols)
    xs_rough = np.linspace(0, dist, mesh_size)
    xs = np.linspace(0, dist, 300)
    bspline = make_interp_spline(xs_rough, tau.numpy())
    ys = bspline(xs)
    ys[0] = 2.0

    if (compute_empirical_cap):
        print("MSD (empirical): ", msd)
        print("Hitting Prob (empirical): ", hits / num_walks)

    np.save("xs_dim"+str(dim), xs)
    np.save("ys_dim"+str(dim), ys)
    return xs, ys

def plot_tau_stats():
    dims = [10, 20, 30, 40, 50]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Isocapacitory Saturation for a Hyperplane")
    ax.set_ylabel("Saturation")
    ax.set_xlabel("Distance from starting point to hyperplane")
    for dim in dims:
        xs, ys = compute_mesh(compute_empirical_vol=True, compute_empirical_cap=False, dist=0.5, dim=dim, mesh_size=7)
        ax.plot(xs, ys, label=str(dim)) 

    labelLines(plt.gca().get_lines(), xvals=(0.2, 0.5), zorder=2.5)
    plt.show()

def main():
    dim = 3 * 32 * 32 #CIFAR10 dim
    alpha = 1.0e0
    num_steps = 100 * alpha
    step = 0.1 / math.sqrt(alpha)
    t = torch.tensor(1.0 / dim)

    sample = torch.zeros(dim)
    msd = .0
    msd_th = math.sqrt(t * dim)
    radius = msd_th

    print("BM run for time: ", t)
    print("RMSD (theoretical): ", msd_th)
    #print("Cap ", compute_cap(dist, compute_empirical_cap))
    #print("Vol ", compute_vol(dist, compute_empirical_vol))

    plot_tau_stats()

if __name__=="__main__":
    main()
