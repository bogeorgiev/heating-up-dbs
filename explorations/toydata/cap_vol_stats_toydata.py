import sys
sys.path.append("../..")
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

from utils.attacks import Adversary
import utils.config as cf
from utils.utils import get_one_vol, get_one_cap

from datasets import Star
from models.toydata.models import MLPModel

import argparse

class Wedge(nn.Module):
    def __init__(self, angle, h, dim=2):
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
        self.normal2.bias.data = self.h
        #print("Wedge normals:")
        #print(self.normal1.weight.data, self.normal1.bias.data)
        #print(self.normal2.weight.data, self.normal2.bias.data)

    def forward(self, x):
        y = self.normal1(x)
        z = self.normal2(x) 
        out = torch.cat([torch.max(-y, torch.zeros(1).to("cuda")),
            torch.max(y, torch.zeros(1).to("cuda"))], dim=1)
        #out = (y > 0) * (z > 0)
        return out 


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

def main():
    parser = argparse.ArgumentParser(description='Stats MLP model')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--dataset-size', type=int, default=200,
                        help='size of train dataset')
    parser.add_argument('--epochs', type=int, default=300000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--enable-logging', type=bool, default=False,
                        help='enables results plotting and tensorboard')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='enable model saving')
    parser.add_argument('--load-model', type=bool, default=False,
                        help='loads a model from model load path')
    parser.add_argument('--model-load-path', type=str,
            default='../../data/saved_models/toydata/mlp_40hu_5l.pt')
    parser.add_argument('--stats-save-path', type=str,
            default='../../data/stats/toydata/')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    star_ds = Star()
    logger = SummaryWriter("./logs/")
    
    loader = torch.utils.data.DataLoader(star_ds,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = MLPModel(hidden_nodes=40).to(device)

    checkpoint = torch.load(args.model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model = Wedge(angle=math.pi, h=-0.5, dim=2).to(device)
    model.eval()

    exp_name = "mlp_40hu_5l"

    dim = 2
    print("Running: ", exp_name)
    dim_sqrt = math.sqrt(dim)
    radius_init = torch.tensor(3.)
    radius_step = torch.tensor(0.01)
    radius_iter = 1000
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

    for i in range(100):
        #data = [torch.zeros(3).float().to(device).unsqueeze(0), torch.zeros(1).long().to(device)] 
        data = next(it)
        x, y = data[0].to(device), data[1].to(device)
        radius = radius_init.clone()
        sigma = radius / dim_sqrt
        vol = 0.
        vol_iter = 5
        r_iter = 0
        while (vol > 0.12) or (vol < 0.10):
            if vol > 0.12:
                radius -= radius_step
            else:
                radius += radius_step

            sigma = radius / dim_sqrt
            vol = 0.
            for j in range(vol_iter):
                vol += get_one_vol(model, x, y, device,
                        radius=radius, num_samples=1000, sample_full_ball=True, shape=[2])
            vol = torch.tensor(vol / vol_iter )
            #print("Vol: ", vol)
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
            np.save(args.stats_save_path + "vol_data_" + exp_name, np.array(vol_data))
            np.save(args.stats_save_path + "cap_data_" + exp_name, np.array(cap_data))
            #np.save("dist_data_2", np.array(dist_data))
            np.save(args.stats_save_path + "radius_data_" + exp_name, np.array(radius_data))

            
        vol_cap = math.pi * ((radius - 0.5)**2) * (2*radius + 0.5) / 3
        vol_ball = 4 * math.pi * (radius**3) / 3
        vol_th = float(vol_cap) / vol_ball
        #print("Dist to Hyperplane ", dist)
        print("Sigma ", sigma)
        print("Radius of Ball ", radius)
        print("Initial Radius ", radius_init)
        print("Vol ", vol) 
        #print("Theory Vol ", vol_th)
        print("Cap ", cap)
        #print("Theory Cap ", 2 * normal_1d.cdf(-0.5 / ( torch.sqrt(t))))
        print("Isoperimetric Bound:", iso_bound)
        print("BM reaches sphere: ", rmsd)
        #print("Mean Dist", dists.mean())
        #print("Dist from center", dist)
        print("----------------------------------")


if __name__=="__main__":
    main()
