import sys
sys.path.append("../../")
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
from models.mnist.models import LeNet5, CNN

from utils.attacks import Adversary
import utils.config as cf
from utils.utils import get_one_vol, get_one_cap
from utils.utils import get_one_sliced_vol, get_one_sliced_cap

def mean_dist(x, radius, dist_samples=500, dist_iter=2):
    # currently investigating between layer one and two
    dim = 32 * 14 * 14
    dim_sqrt = math.sqrt(dim)
    sigma = radius / dim_sqrt
    dists = torch.zeros(dist_samples).float().to(device)
    for i in range(dist_iter):
        x_exp_orig = x.repeat(dist_samples, 1, 1, 1)
        y_exp = y.repeat(dist_samples)
        #x_exp = x_exp_orig + torch.randn_like(x_exp_orig) * sigma
        #print((x_exp-x_exp_orig).norm(dim=1).norm(dim=1).norm(dim=1))
        #dists += adv.get_distances(model, x_exp, y_exp, device, eps=1.0, alpha=1.0e-3)[0]
        
        # after the first layer
        pert = torch.randn(dist_samples, 32, 14, 14).to(device) * sigma
        dists += adv.get_sliced_distances(
                model, x_exp_orig, pert, y_exp, device, eps=1.0, alpha=1.0e-3)[0]

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
            #transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
            ])
    num_examples = 400
    batch_size = 1
    dataset = torchvision.datasets.MNIST(root='../../data/datasets/mnist/', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_path = str("../../models/mnist/trained_models/normal_LeNet.pth")

    device = "cuda"
    torch.cuda.set_device(0)
    model = LeNet5().to(device)
    """
    dimension at intermediate layers , x = num_examples
    input:            x 1 28 28
    maxpool1:         x 32 14 14
    maxpool2:         x 64 7 7
    lin1+relu:        x 200
    lin2/output       x 10
    """
    #model = CNN.to(device)
    model.load_state_dict(torch.load(model_path))
    #model = nn.DataParallel(model)
    model.eval()
    
    # layer dependent dimension
    dim = 200
    pert = torch.zeros(1, 200)
    
    dim_sqrt = math.sqrt(dim)
    radius_init = torch.tensor(5.)
    radius_step = torch.tensor(0.1)
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
    tau_data = []
    dist_data = []
    radius_data = []

    for i in range(num_examples):
        data = next(it)
        x, y = data[0].to(device), data[1].to(device)
        radius = radius_init.clone()
        sigma = radius / dim_sqrt
        vol = 0.
        vol_iter = 3
        r_iter = 0

        upper = 0.011
        lower = 0.009
        
        while (vol > upper) or (vol < lower):
            if vol > upper:
                radius -= radius_step
            else:
                radius += radius_step

            sigma = radius / dim_sqrt
            vol = 0.
            for j in range(vol_iter):
                vol += get_one_sliced_vol(model, x, pert, y, device,
                            radius=radius, num_samples=600, sample_full_ball=True)
            vol = torch.tensor(vol / vol_iter )
            r_iter += 1
            if r_iter > radius_iter:
                break
        #dists = mean_dist(x, radius)
 
        dist = adv.get_sliced_distances(
                model, x, pert, y, device, eps=5.0e-1, alpha=1.0e-3, max_iter=10000)[0]

        step = 1.0e-1 * radius / (dim_sqrt * math.sqrt(alpha))
        t = torch.tensor(num_steps * step**2)
        rmsd = torch.sqrt(dim * t)
        cap = 0.
        cap_iter = 2
        for j in range(cap_iter):
            cap += get_one_sliced_cap(model, x, pert, y, device,
                        step=step, num_steps=num_steps, num_walks=1000, j="")
        cap = torch.tensor(cap / cap_iter )

        iso_bound = 0
        if vol < 0.5:
            iso_bound = -sigma * normal_1d.icdf(vol)

        vol_data += [vol.data]
        cap_data += [cap.data]
        tau_data += [cap.data/vol.data]
        dist_data += [dist.cpu().detach().numpy()]
        radius_data += [radius.data]
        
        """
        if (i+1) % 5 == 0:
          i  np.save("../../data/cap_vol_stats_mnist/vol_data_untrained_LeNet", np.array(vol_data))
            np.save("../../data/cap_vol_stats_mnist/cap_data_untrained_LeNet", np.array(cap_data))
            #np.save("dist_data_2", np.array(dist_data))
            np.save("../../data/cap_vol_stats_mnist/radius_LeNet", np.array(radius_data))
        """

        #print("Dist to Hyperplane ", dist)
        print("Sigma ", sigma)
        print("Radius of Ball ", radius)
        print("Initial Radius ", radius_init)
        print("Vol ", vol) 
        print("Cap ", cap)
        print("Tau", cap/vol)
        print("Normed dist", dist.cpu().detach().numpy())
        print("Isoperimetric Bound:", iso_bound)
        print("BM reaches sphere: ", rmsd)
        #print("Mean Dist", dists.mean())
        #print("Dist from center", dist)
        print("----------------------------------")

    f = open('saves/cap_vol_tau_sliced_layer34_LeNet.txt', 'w')
    for i in range(num_examples):
        f.write(str(i+1)+' '+
                str(cap_data[i].item())+' '+
                str(vol_data[i].item())+' '+
                str(tau_data[i].item())+' '+
                str(radius_data[i].item())+' '+
                str(dist_data[i].item())+'\n'
                )
    f.close()




