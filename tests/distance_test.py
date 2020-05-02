import torch
import torchvision
import torchvision.transforms as transforms
import sys
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
num_examples = 10000
batch_size = 100
eps = 1.
adv = Adversary('fgsm', torch.device('cuda:0'))
model_address = '../models/cifar10/trained_models/adversarial_fgsm_WideResNet_28_10_run_1.pth'
model = Wide_ResNet(28, 10, 0.3, 10).to(device)
model.load_state_dict(torch.load(model_address))
model.eval()

transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
                ])

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=num_examples, shuffle=False)
it = iter(loader)
data = next(it)
x, y = data[0].to(device), data[1].to(device)

steps = 1
distances = []

for i in range(int(num_examples/batch_size)):
    hold, _ = adv.get_distances(model, x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size], device, eps=eps)
    distances += hold.tolist()
    print('in eps {} iteration {}/{} with mean {}'.format(eps, i+1, int(num_examples/batch_size), mean(distances[-batch_size:])))

avg_distances = mean(distances)

f = open('saves/distances.txt', 'w')
for i in range(steps*num_examples):
    f.write(str(eps)+' '+str(distances[i])+'\n')
f.write('\n'+'mean:\n')
f.write(str(avg_distances)+'\n')
f.close

print('Experiment conducted, find results in saves!')
