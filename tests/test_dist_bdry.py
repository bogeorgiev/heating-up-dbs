import sys
sys.path.append("../")
import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models import Wide_ResNet

from attacks import Adversary
import config as cf

if __name__=="__main__":
    print("Test Distance")

    transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
            ])

    batch_size = 1
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model_path = str("../cifar_model_saves/noisy_0.4_WideResNet_28_10_run_1.pth")

    device = "cuda"
    model = Wide_ResNet(28, 10, 0.3, 10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    it = iter(loader)
    data = next(it)
    x, y = data[0].to(device), data[1].to(device)

    print(model(x))

    adv = Adversary("pgd_linf", device)
    print(adv.get_distances(model, x, y, device, eps=0.3))


