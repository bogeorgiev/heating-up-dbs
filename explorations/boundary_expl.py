import torch
import numpy as np
import random as rd
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models import Wide_ResNet

from utils import *
from attacks import Adversary
import config as cf



class Experiment:
    def __init__(self, num_examples, const, dim, num_rw):
        
        # concerning data
        self.num_examples = num_examples
        self.const = const
        self.cap_train = None
        self.cap_test = None
        self.vol_train = None
        self.vol_test = None
        self.spik_train = np.ones(num_examples) * (-1.)
        self.spik_test = np.ones(num_examples) * (-1.)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #transform = transforms.Compose([transforms.ToTensor(), 
        #                                     transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
        #                                     ])
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
        data = next(it)
        x, y = data[0], data[1]
        self.x_test, self.y_test = Variable(x.to(self.device)), Variable(y.to(self.device))

        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=num_examples, shuffle=False)
        it = iter(loader)
        data = next(it)
        data = next(it)
        x, y = data[0], data[1]
        print(x)
        self.x_train, self.y_train = Variable(x.to(self.device)), Variable(y.to(self.device))

        # concerning the model
        self.model = None

        # concerning run params
        self.num_walks = num_rw
        self.const = const
        self.dim = dim
        self.num_steps = 500
        self.num_samples = 100
        self.assisstant = Adversary('pgd_linf', self.device)
        # neglect examples where no adv example was found; 1 if adv examples was found
        self.distances_train, self.neglect_train = None, None
        self.distances_test, self.neglect_test = None, None
        self.step_test = None
        self.t_test = None
        self.step_train = None
        self.t_train = None
        self.radius_train = None
        self.radius_test = None
        # concerning experiment framework
        self.save_file = None
        self.save_folder = 'save_explorations/'
        self.run_id = rd.randint(0, 1e6)

    def load_model(self, model, training):    
        # concerning the model
        self.eps = 0.01
        self.model = Wide_ResNet(28, 10, 0.3, 10).to(self.device)
        self.model.load_state_dict(torch.load(model))
        print('loaded model for: ' + training)
        self.model.eval()
        print('model evaluated!')
        self.save_file = 'cap_vol_spi_'+training+str(self.run_id)+'.txt'
        self.save_folder = 'save_explorations/'
        
        # neglect examples where no adv example was found; 1 if adv examples was found
        self.distances_train, self.neglect_train = self.assisstant.get_distances(
                self.model, self.x_train, self.y_train, self.device, eps=self.eps)
        self.distances_test, self.neglect_test = self.assisstant.get_distances(
                self.model, self.x_test, self.y_test, self.device, eps=self.eps)
        
        self.step_test = self.distances_test / self.const / np.sqrt(self.dim) / np.sqrt(self.num_steps)
        self.t_test = self.num_steps * self.step_test**2
        self.step_train = self.distances_train / self.const / np.sqrt(self.dim) / np.sqrt(self.num_steps)
        self.t_train = self.num_steps * self.step_train**2
        print(self.neglect_train)
        print(self.neglect_test)
        print('distances train: ', self.distances_train)
        print('distances test: ', self.distances_test)
        print('num steps: ', self.num_steps)
        print('step_train:', self.step_train)
        print('step_test: ', self.step_test)
        print('t_train: ', self.t_train)
        print('t_test: ', self.t_test)
        self.radius_train = np.sqrt(self.t_train.detach().cpu().numpy()) * np.sqrt(self.dim)
        print('radius 1: ', self.radius_train)
        self.radius_test = np.sqrt(self.t_test.detach().cpu().numpy()) * np.sqrt(self.dim)
        print('radius 2: ', self.radius_test)

        
    def conduct(self, model, training):
        self.load_model(model, training)
        print('commencing evaluation...')
        
        self.cap_train = get_caps(self.model, self.x_train, self.y_train, 
                                  self.device, self.step_train, self.num_steps, self.num_walks)
        print(self.cap_train)
        self.vol_train = get_vols(self.model, self.x_train, self.y_train,
                                  self.device, self.radius_train, self.num_samples)
        print(self.vol_train)
        #self.spik_train[self.vol_train != torch.zeros(self.num_examples)] = np.array(self.cap_train) / np.array(self.vol_train)
        for i in range(self.num_examples):
            if self.vol_train[i] != 0.:
                self.spik_train[i] = self.cap_train[i] / self.vol_train[i]

        print(self.spik_train)     
        self.cap_test = get_caps(self.model, self.x_test, self.y_test, 
                                  self.device, self.step_test, self.num_steps, self.num_walks)
        print(self.cap_train)
        self.vol_test = get_vols(self.model, self.x_test, self.y_test,
                                  self.device, self.radius_test, self.num_samples)
        print(self.vol_test)
        #self.spik_test[self.vol_test != torch.zeros(self.num_examples)] = np.array(self.cap_test) / np.array(self.vol_test)
        for i in range(self.num_examples):
            if self.vol_test[i] != 0.:
                self.spik_test[i] = self.cap_test[i] / self.vol_test[i]
        print(self.spik_test)
        print('evalutation finished, find results in '+self.save_file)
        self.save_results()
        print('Experiment finished!')

        
    def save_results(self):
        f = open(self.save_folder+self.save_file, 'w')
        for i in range(self.num_examples):
            f.write('train '+
                    str(self.neglect_train[i].item())+' '+
                    str(self.distances_train[i].item())+' '+
                    str(self.cap_train[i])+' '+
                    str(self.vol_train[i])+' '+
                    str(self.spik_train[i])+' '+
                    str(self.step_train[i])+' '+
                    str(self.radius_train[i])+' '+
                    '\n')
        for i in range(self.num_examples):
            f.write('test '+
                    str(self.neglect_test[i].item())+' '+
                    str(self.distances_test[i].item())+' '+
                    str(self.cap_test[i])+' '+
                    str(self.vol_test[i])+' '+
                    str(self.spik_test[i])+' '+ 
                    str(self.step_test[i])+' '+
                    str(self.radius_test[i])+' '+
                    '\n')
        f.write('experiment_data: '+str(self.num_steps)+' '+str(self.const))
        f.close()




exp = Experiment(10, 0.013, 3.*32.*32., 100)
#exp.conduct('cifar_model_saves/normal_WideResNet_28_10_run_1.pth', 'normal')
exp.conduct('cifar_model_saves/noisy_0.4_WideResNet_28_10_run_1.pth', 'noisy_04')
#exp.conduct('cifar_model_saves/noisy_0.1_WideResNet_28_10_run_1.pth', 'noisy_01')
#exp.conduct('cifar_model_saves/adversarial_fgsm_WideResNet_28_10_run_1.pth', 'adversarial')
