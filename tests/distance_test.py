import torch
import torchvision
import torchvision.transforms as transforms
import sys
import numpy as np
sys.path += ['../']
sys.path += ['../utils/']
from models.cifar10.models import Wide_ResNet
import config as cf
from attacks import *

def mean(liste):
    summe = 0.
    for el in liste:
        summe += el
    return summe/len(liste)


device = torch.device("cuda:0")
num_examples = 1000
batch_size = 100
eps = 1.
alpha = 1e-3
max_iter = 10000
steps = np.zeros(num_examples)
adv = Adversary('fgsm', torch.device('cuda:0'))
model_address = '../models/cifar10/trained_models/noisy_0.4_WideResNet_28_10_run_1.pth'
model = Wide_ResNet(28, 10, 0.3, 10).to(device)
model.load_state_dict(torch.load(model_address))
model.eval()

transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
                ])

dataset = torchvision.datasets.CIFAR10(root='../data/datasets/cifar10', train=False, download=False, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=num_examples, shuffle=False)
it = iter(loader)
data = next(it)
x, y = data[0].to(device), data[1].to(device)

distances = []
step_counter = []

for i in range(int(num_examples/batch_size)):
    hold, _, steps = adv.get_distances(model, x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size], device, 
                                eps=eps, alpha=alpha, max_iter=max_iter)
    distances += hold.tolist()
    step_counter += steps.tolist()
    print('in eps {} iteration {}/{} with mean {}'.format(eps, i+1, int(num_examples/batch_size), mean(distances[-batch_size:])))

print(distances)
print(step_counter)

avg_distances = mean(distances)

f = open('saves/distances.txt', 'w')
for i in range(num_examples):
    f.write(str(eps)+' '+str(distances[i])+' '+str(step_counter[i])+'\n')
f.close()

print('Experiment conducted, find results in saves!')
