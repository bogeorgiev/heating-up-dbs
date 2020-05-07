import sys
sys.path.append("../")
import math
import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from models.cifar10.models import Wide_ResNet

from utils.attacks import Adversary
import utils.config as cf
from utils.utils import get_one_vol, get_one_cap


def mean_dist(x, radius, dist_samples=500, dist_iter=2):
    dim = 3 * 32 * 32
    dim_sqrt = math.sqrt(dim)
    sigma = radius / dim_sqrt
    dists = torch.zeros(dist_samples).float().to(device)
    for i in range(dist_iter):
        x_exp_orig = x.repeat(dist_samples, 1, 1, 1)
        y_exp = y.repeat(dist_samples)
        x_exp = x_exp_orig + torch.randn_like(x_exp_orig) * sigma
        #print((x_exp-x_exp_orig).norm(dim=1).norm(dim=1).norm(dim=1))
        dists += adv.get_distances(model, x_exp, y_exp, device, eps=1.0, alpha=1.0e-3)[0]
        #dist = adv.get_distances(model, x, y, device, eps=1.0e-1, alpha=1.0e-3)[0]
    dists /= dist_iter
    return dists


if __name__=="__main__":
    print("Test Distance")

    torch.manual_seed(0)
    transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
            ])
    batch_size = 1
    dataset = torchvision.datasets.CIFAR10(root='../../../cap-vol-analysis/cap-vol-analysis/data/datasets/cifar10/', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model_path = str("../../../cap-vol-analysis/cap-vol-analysis/data/saved_models/cifar10/noisy_0.1_WideResNet_28_10_run_1.pth")

    device = "cuda"
    torch.cuda.set_device(0)
    model = Wide_ResNet(28, 10, 0.3, 10).to(device)
    model.load_state_dict(torch.load(model_path))
    model = nn.DataParallel(model)
    model.eval()

    dim = 3 * 32 * 32
    dim_sqrt = math.sqrt(dim)
    radius_init = torch.tensor(38.)
    radius_step = torch.tensor(2.)
    radius_iter = 100
    radius = torch.tensor(40.)
    alpha = 2.0e0
    num_steps = int(100 * alpha)
    step = 1.0e-1 * radius / (dim_sqrt * math.sqrt(alpha))
    c = 0.1
    
    normal_1d = Normal(torch.tensor(0.0), torch.tensor(1.0))

    t = torch.tensor(num_steps * step**2)
    rmsd = torch.sqrt(dim * t)
    #dist = c * rmsd

    print("Runtime: ", t)
    print("RMSD: ", rmsd)
    #print("Dist to hyperplane", dist)

    it = iter(loader)
    adv = Adversary("pgd_linf", device)
    
    vol_data = []
    cap_data = []
    dist_data = []
    radius_data = []

    for i in range(1000):
        data = next(it)
        x, y = data[0].to(device), data[1].to(device)
        radius = radius_init.clone()
        sigma = radius / dim_sqrt
        vol = 0.
        vol_iter = 3
        r_iter = 0
        while (vol > 0.3) or (vol < 0.1):
            if vol > 0.3:
                radius -= radius_step
            else:
                radius += radius_step

            sigma = radius / dim_sqrt
            vol = 0.
            for j in range(vol_iter):
                vol += get_one_vol(model, x, y, device,
                            radius=radius, num_samples=600, sample_full_ball=True)
            vol = torch.tensor(vol / vol_iter )
            r_iter += 1
            if r_iter > radius_iter:
                break
        #dists = mean_dist(x, radius)

        #dist = adv.get_distances(model, x, y, device, eps=1.0e-1, alpha=1.0e-3, max_iter=100)[0]

        step = 1.0e-1 * radius / (dim_sqrt * math.sqrt(alpha))
        t = torch.tensor(num_steps * step**2)
        rmsd = torch.sqrt(dim * t)
        cap = 0.
        cap_iter = 2
        for j in range(cap_iter):
            cap += get_one_cap(model, x, y, device,
                        step=step, num_steps=num_steps, num_walks=500, j="")
        cap = torch.tensor(cap / cap_iter )

        iso_bound = 0
        if vol < 0.5:
            iso_bound = -sigma * normal_1d.icdf(vol)

        vol_data += [vol.data]
        cap_data += [cap.data]
        #dist_data += [dist.cpu().detach().numpy()]
        radius_data += [radius.data]
        
        if (i+1) % 5 == 0:
            np.save("vol_data_noisy0.1", np.array(vol_data))
            np.save("cap_data_noisy0.1", np.array(cap_data))
            #np.save("dist_data_2", np.array(dist_data))
            np.save("radius_data_0.1", np.array(radius_data))

        #print("Dist to Hyperplane ", dist)
        print("Sigma ", sigma)
        print("Radius of Ball ", radius)
        print("Initial Radius ", radius_init)
        print("Vol ", vol) 
        print("Cap ", cap)
        print("Isoperimetric Bound:", iso_bound)
        print("BM reaches sphere: ", rmsd)
        #print("Mean Dist", dists.mean())
        #print("Dist from center", dist)
        print("----------------------------------")



