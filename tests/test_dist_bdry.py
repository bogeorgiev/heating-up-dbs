import sys
sys.path.append("../")
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

    torch.manual_seed(0)
    transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
            ])
    batch_size = 1
    dataset = torchvision.datasets.CIFAR10(root='../data/datasets/cifar10/', train=False, download=False, transform=transform)
    #loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model_path = str("../data/saved_models/cifar10/noisy_0.1_WideResNet_28_10_run_1.pth")

    device = "cuda"
    torch.cuda.set_device(0)
    model = Wide_ResNet(28, 10, 0.3, 10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #it = iter(loader)
    data = dataset[0] #next(it)
    x, y = data[0].to(device).unsqueeze(0), torch.tensor(data[1]).to(device).unsqueeze(0)
    print(model(x))

    adv = Adversary("pgd_linf", device)
    print(adv.get_distances(model, x, y, device, eps=1.0e-1, alpha=1.0e-3))

    vols = 0.
    for i in range(100):
        vols += (get_one_vol(model, x, y, device, radius=20, num_samples=200)) 
    print(vols)
