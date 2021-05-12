import sys
sys.path.append("../../")
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models.mnist.models import LeNet5
from utils.attacks import Adversary


device = 'cuda'
num_examples = 1000

model_name = [
        'normal',
        'adv_fgsm',
        'noisy 04',
        'adv_random_1',
        'adv_random_2',
        'adv_random_4'
        ]
models = [
        '../../train/mnist/trained_models/normal_LeNet.pth',
        '../../train/mnist/trained_models/adversarial_fgsm_LeNet.pth',
        '../../train/mnist/trained_models/noisy_0.4_LeNet.pth',
        '../../train/mnist/trained_models/adversarial_random_1_ex_LeNet.pth',
        '../../train/mnist/trained_models/adversarial_random_2_ex_LeNet.pth',
        '../../train/mnist/trained_models/adversarial_random_4_ex_LeNet.pth',
        ]

f = open('saves/Lenet_robustness_random_adv_models.txt', "w")

for path, name in zip(models, model_name):
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    # test accuracy on clean dataset
    
    transform = transforms.ToTensor()
    batch_size = 200
    dataset = torchvision.datasets.MNIST(
            root='../../data/datasets/mnist/', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    acc = 0.
    total = 0.
    for _, (x, y) in enumerate(loader):
        x, y = Variable(x).to(device), Variable(y).to(device)
        outcome = model(x)
        _, pred = torch.max(outcome.data, 1)
        acc += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        total += y.size(0)
    acc = acc/total
    print('clean data accuracy for '+name+': ', acc)
    acc_clean = acc

    # test accuary during pgd attack:  epsilon = 0.5

    batch_size = 100
    dataset = torchvision.datasets.MNIST(
            root='../../data/datasets/mnist/', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    adversary = Adversary('pgd', device)
    epsilon = 0.5
    step_size = epsilon / 25.
    num_steps = 100
    acc = 0.
    total = 0.
    for _, (x, y) in enumerate(loader):
        x, y = Variable(x).to(device), Variable(y).to(device)
        delta = adversary.pgd(model, x, y, epsilon, step_size, num_steps).to(device)
        outcome = model(x+delta)
        _, pred = torch.max(outcome, 1)
        acc += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        total += y.size(0)
    acc = acc/total
    print('epsilon 0.5 pgd attack accuracy for '+name+': ', acc)
    acc_pgd_05 = acc

    # test accuary during pgd attack:  epsilon = 1.

    batch_size = 100
    dataset = torchvision.datasets.MNIST(
            root='../../data/datasets/mnist/', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    adversary = Adversary('pgd', device)
    epsilon = 1.0
    step_size = epsilon / 25.
    num_steps = 100
    acc = 0.
    total = 0.
    for _, (x, y) in enumerate(loader):
        x, y = Variable(x).to(device), Variable(y).to(device)
        delta = adversary.pgd(model, x, y, epsilon, step_size, num_steps).to(device)
        outcome = model(x+delta)
        _, pred = torch.max(outcome, 1)
        acc += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        total += y.size(0)
    acc = acc/total
    print('epsilon 1.0 pgd attack accuracy for '+name+': ', acc)
    acc_pgd_1 = acc

    # test accuary during gaussian perturbation:  variance = 0.4

    batch_size = 100
    dataset = torchvision.datasets.MNIST(
            root='../../data/datasets/mnist/', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    var = 0.4
    acc = 0.
    total = 0.
    for _, (x, y) in enumerate(loader):
        x, y = Variable(x).to(device), Variable(y).to(device)
        x += torch.randn(x.size()).to(device) * var
        outcome = model(x+delta)
        _, pred = torch.max(outcome, 1)
        acc += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        total += y.size(0)
    acc = acc/total
    print('var 0.4 gaussian perturbation on '+name+': ', acc)
    acc_var_04 = acc

    # test accuary during fog corruption:

    batch_size = 100
    dataset = torchvision.datasets.MNIST(
            root='../../data/datasets/mnist/', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    severity = 4
    acc = 0.
    total = 0.
    for _, (x, y) in enumerate(loader):
        fog = adversary.fog(x, severity=severity).to(device)
        x = x.to(device)
        x, y = Variable(x+fog).to(device), Variable(y).to(device)
        x += torch.randn(x.size()).to(device) * var
        outcome = model(x+delta)
        _, pred = torch.max(outcome, 1)
        acc += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        total += y.size(0)
    acc = acc/total
    print('fog corruption (Gilmer: MNIST-C) on '+name+': ', acc)
    acc_fog = acc
    
    f.write("for model "+name+':\n')
    f.write('clean data: '+str(acc_clean)+'\n')
    f.write('pdg attack 0.5: '+str(acc_pgd_05)+'\n')
    f.write('pgd_attack 1.0: '+str(acc_pgd_1)+'\n')
    f.write('gaussian noise 0.4 var: '+str(acc_var_04)+'\n')
    f.write('fog corruption: '+str(acc_fog)+"\n")
f.close()

