import sys
sys.path.append("../")
import math
import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.cifar10.models import Wide_ResNet

from utils.attacks import Adversary
import utils.config as cf
from utils.utils import get_one_vol

if __name__=="__main__":
    print("Test Distance")

    #torch.manual_seed(0)
    transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
            ])
    batch_size = 1
    dataset = torchvision.datasets.CIFAR10(root='../../../cap-vol-analysis/cap-vol-analysis/data/datasets/cifar10/', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model_path = str("../../../cap-vol-analysis/cap-vol-analysis/data/saved_models/cifar10/normal_WideResNet_28_10_run_1.pth")

    device = "cuda"
    torch.cuda.set_device(0)
    model = Wide_ResNet(28, 10, 0.3, 10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dim = 3 * 32 * 32
    dim_sqrt = math.sqrt(dim)
    num_steps = 500
    step = 1.0e-1
    c = 0.1

    #t = torch.tensor(num_steps * step**2)
    #rmsd = torch.sqrt(dim * t)
    #dist = c * rmsd

    #print("Runtime: ", t)
    #print("RMSD: ", rmsd)
    #print("Dist to hyperplane", dist)

    it = iter(loader)
    adv = Adversary("pgd_linf", device)

    for i in range(100):
        data = next(it)
        x, y = data[0].to(device), data[1].to(device)
        radius = torch.tensor(10.) #dist / c
        sigma = radius / dim_sqrt
        x_exp_orig = x.repeat(50, 1, 1, 1)
        y_exp = y.repeat(50)
        x_exp = x_exp_orig + torch.randn_like(x_exp_orig) * sigma
        print("Sigma ", sigma)
        #print((x_exp-x_exp_orig).norm(dim=1).norm(dim=1).norm(dim=1))
        #dists = adv.get_distances(model, x_exp, y_exp, device, eps=1.0e-1, alpha=1.0e-3)[0]
        #dist = adv.get_distances(model, x, y, device, eps=1.0e-1, alpha=1.0e-3)[0]
        #print("Mean Dist", dists.mean())
        #print("Dist from center", dist)
        vol = 0.
        vol_iter = 100
        for j in range(vol_iter):
            vol += get_one_vol(model, x, y, device,
                        radius=sigma, num_samples=200)
        vol = vol / vol_iter 
        #print("Dist to Hyperplane ", dist)
        print("Radius of Ball ", radius)
        print("Vol ", vol)
        print("----------------------------------")
