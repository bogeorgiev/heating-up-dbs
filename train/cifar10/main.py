import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import time
import sys
import numpy as np
import math
import config as cf

from attacks import Adversary

# MNIST models
from models import LeNet5, DNN2, DNN4, CNN
# CIFAR10 models
from models import Wide_ResNet

from utils import Savestate
from utils import *


class Experiment:
    def __init__(self, cf):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_batch_size = cf.train_batch_size 
        self.test_batch_size = cf.test_batch_size
        self.num_epochs = cf.num_epochs
        self.curr_epoch = None
        self.curr_batch = None
        self.learning_rate = cf.lr
        self.weight_decay = cf.weight_decay
        self.num_runs = cf.num_runs
        self.training_type = cf.training_type
        self.adversary = Adversary(cf.attack, self.device)
        self.variance = cf.variance
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None

        self.save_quantities = Savestate(cf.save_file_quants)
        self.save_file_model = cf.save_file_model
        self.writer = None
        self.save_path_model = cf.save_path_model
        
        self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
                ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
            ])


    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path_model+self.save_file_model+str(self.curr_run)+'.pth')


    def conduct(self):
        for self.curr_run in range(1, self.num_runs+1):
            self.writer = SummaryWriter()
            self.reset()
            self.train()
            self.writer.close()
            self.save_model()
            print('run {} finished!'.format(self.curr_run))
        # self.save_quantities.write_in_file()
        print('experiment finished!')#; find results in ' + self.save_quantities.filename)
        return
    
    
    def reset(self):
        self.load_model()
        self.load_data()
        self.curr_epoch = None
        self.curr_batch = None


    def load_data(self):
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform_train)     
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=self.transform_test)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=100, shuffle=False, num_workers=2)


    def load_model(self):
        # MNIST
        # self.model = LeNet5().to(self.device)
        self.model = Wide_ResNet(28, 10, 0.3, 10).to(self.device)
        test = self.model(Variable(torch.randn(1,3,32,32).to(self.device)))
        print('test size: ', test.size())
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        elapsed_time = 0

        for self.curr_epoch in range(1, self.num_epochs+1):
            self.model.train()
            self.model.training = True
            self.optimizer = optim.SGD(self.model.parameters(), 
                lr=cf.learning_rate(self.learning_rate, self.curr_epoch), momentum=0.9, weight_decay=5e-4)
            train_loss = 0
            train_correct = 0
            total = 0
            time_start = time.time()

            print('\n=> Training Epoch #%d, LR=%.4f' %(self.curr_epoch, cf.learning_rate(self.learning_rate, self.curr_epoch)))
            for self.curr_batch, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                # perturb data during noisy training
                if self.training_type == 'noisy':
                    x = self.adversary.perturb(x, self.device, self.variance)
                x, y = Variable(x), Variable(y)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                total += y.size(0)
                loss = self.criterion(outputs, y)
                train_loss += loss
                _, pred = torch.max(outputs.data, 1)
                train_correct += pred.eq(y.data).cpu().sum()
                loss.backward()
                self.optimizer.step()

                # add training on adversarial perturbation during adv training
                if self.training_type == 'adversarial':
                    delta = self.adversary.get_adversarial_examples(
                            self.model, x, y).to(self.device)
                    x, y = x.to(self.device), y.to(self.device)
                    x, y = Variable(x), Variable(y)
                    outcome = self.model(x+delta)
                
                    _, pred = torch.max(outcome.data, 1)
                    train_correct += pred.eq(y.data).cpu().sum()
                    total += y.size(0)
                    loss = self.criterion(outcome, y)
                    train_loss += loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(self.curr_epoch, self.num_epochs, self.curr_batch,
                    (len(self.train_dataset)//self.train_batch_size)+1, train_loss.item(), 100.*train_correct/total))
                sys.stdout.flush()            

            train_acc = 100.*train_correct/total
            
            with torch.no_grad():
                # testing
                self.model.eval()
                self.training = False
                test_loss = 0.
                test_correct = 0
                total = 0
                for self.curr_batch, (x, y) in enumerate(self.test_loader):
                    x_var, y_var = Variable(x), Variable(y)
                    x_var, y_var = x_var.to(self.device), y_var.to(self.device)
                    outcome = self.model(x_var)
                    loss = self.criterion(outcome, y_var)
                    test_loss += loss
                    _, pred = torch.max(outcome.data, 1)
                    test_correct += pred.eq(y_var.data).cpu().sum()
                    total += y_var.size(0)
           
                test_acc = 100.*test_correct/total
                print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(self.curr_epoch, test_loss.item(), test_acc))
            
            time_epoch = time.time() - time_start
            elapsed_time += time_epoch
            print('| Elapsed time : %d:%02d:%02d' %(cf.get_hms(elapsed_time)))
            self.write_tb(train_loss.item(), train_acc, test_loss.item(), test_acc)

            #if self.curr_epoch == self.num_epochs:
            #    spikiness, avg_cap, avg_vol = get_spikiness(
            #            self.model, self.spikiness_dataset, self.device,self.adversary)
            #    print('final spikiness {}, avg_cap {}, avg_vol {}'.format(spikiness, avg_cap, avg_vol))
            #else: 
            #    spikiness, avg_cap, avg_vol = -1., -1., -1.
            
            #print('epoch {}/{}, train loss: {}, val loss: {}, spikiness: {}'.format(
              #  self.curr_epoch, self.num_epochs, train_loss, test_loss, spikiness))
            # print('avg cap: {}, avp vol: {}'.format(avg_cap, avg_vol))
            #print('epoch {}/{}, train loss: {}, val loss: {}'.format(
            #    self.curr_epoch, self.num_epochs, train_loss, test_loss))
            
            #print('train correct: {}, test correct: {}'.format(train_correct, test_correct))
            #self.save(train_loss, train_correct.item(), test_loss, test_correct.item(), spikiness)
            #self.write_tb(train_loss, train_correct, test_loss, test_correct, spikiness, avg_cap, avg_vol)

        
    def save(self, train_loss, train_class, test_loss, test_class, spikiness):
        self.save_quantities.run += [self.curr_run]
        self.save_quantities.epoch += [self.curr_epoch]
        self.save_quantities.train_loss += [train_loss]
        self.save_quantities.train_class += [train_class]
        self.save_quantities.test_loss += [test_loss]
        self.save_quantities.test_class += [test_class]
        self.save_quantities.spikiness += [spikiness]


    def write_tb(self, train_loss, train_correct, test_loss, test_correct):#, spikiness, avg_cap, avg_vol):
        self.writer.add_scalar('Loss/train', train_loss, self.curr_epoch)
        self.writer.add_scalar('Loss/test', test_loss, self.curr_epoch)
        self.writer.add_scalar('Accuracy/train', train_correct, self.curr_epoch)
        self.writer.add_scalar('Accuracy/test', test_correct, self.curr_epoch)
        #self.writer.add_scalar('Spikiness/spikiness', spikiness, self.curr_epoch)
        #self.writer.add_scalar('Spikiness/avg capacity', avg_cap, self.curr_epoch)
        #self.writer.add_scalar('Spikiness/avg volume', avg_vol, self.curr_epoch)


experiment = Experiment(cf)
experiment.conduct()


