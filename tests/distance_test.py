import torch
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import sys
import numpy as np
sys.path += ['../']
sys.path += ['../utils/']
from models.cifar10.models import Wide_ResNet, resnet110
import config as cf
from attacks import *
from torch.nn.parallel import DistributedDataParallel as DDP

def mean(liste):
    summe = 0.
    for el in liste:
        summe += el
    return summe/len(liste)


device = torch.device("cuda")
num_examples = 1000
batch_size = 100
eps = 1.
alpha = 1e-4
max_iter = 100000
steps = np.zeros(num_examples)
adv = Adversary('fgsm', device)
model_address = '../models/cifar10/trained_models/normal_WideResNet_28_10_run_1.pth'
#model_address = '../models/cifar10/trained_models/ResNet_110.th'


state_dict = torch.load(model_address)

"""
####### the following is for ResNet loading only #####
new_state_dict = OrderedDict()
for k, v in state_dict['state_dict'].items():
    name = k[7:] 
    new_state_dict[name] = v
######################################################
"""

model = Wide_ResNet(28, 10, 0.3, 10).to(device)
#model = resnet110().to(device)
#model.load_state_dict(torch.load(model_address))
model.load_state_dict(state_dict)
#model.load_state_dict(new_state_dict)
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
    print('recently found distances: ', hold.tolist())
    print('in eps {} iteration {}/{} with mean {}'.format(eps, i+1, int(num_examples/batch_size), mean(distances[-batch_size:])))

print(distances)
print(step_counter)

avg_distances = mean(distances)

f = open('saves/distances.txt', 'w')
for i in range(num_examples):
    f.write(str(eps)+' '+str(distances[i])+' '+str(step_counter[i])+'\n')
f.close()

print('Experiment conducted, find results in saves!')
