import sys
sys.path.append('../../models/mnist/')
sys.path.append('../../utils/')
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import time
import numpy as np

from attacks import Adversary

from models import LeNet5, DNN2, DNN4, CNN

# parameters
param = {
    'train_batch_size': 64,
    'test_batch_size': 100,
    'num_epochs': 50,
    'delay': 10,
    'learning_rate': 1e-3,
    'weight_decay': 5e-4,
    'num_runs': 1,
    #training types: 'normal', 'noisy', 'adversarial'
    'training_type': 'adversarial',
    'save_filename': 'save_results.txt',
    # available strategies: 'fgsm', 'pgd', 'pgd_linf', 'pgd_linf_rand'
    'attack': 'random_walk',
    # variance for noisy training
    'variance': 0.8
}


class Experiment:
    def __init__(self, param):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_batch_size = param['train_batch_size']
        self.test_batch_size = param['test_batch_size']
        self.num_epochs = param['num_epochs']
        self.curr_epoch = None
        self.curr_batch = None
        self.learning_rate = param['learning_rate']
        self.weight_decay = param['weight_decay']
        self.num_runs = param['num_runs']
        self.training_type = param['training_type']
        # attacks and data; also provides method 'perturb' to add noise to data
        self.adversary = Adversary(param['attack'], self.device)
        self.adv_num_examples = 4
        self.adv_step_size = 0.25
        self.adv_max_iter = 100
        self.attack = param['attack']
        self.variance = param['variance']
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
 #       self.writer = None
        self.model_save_path = './trained_models/'


    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path+'adversarial_random_'+str(self.adv_num_examples)+'_ex_LeNet.pth')


    def conduct(self):
        for self.curr_run in range(1, self.num_runs+1):
            self.reset()
            self.train()
            self.save_model()
            print('run {} finished!'.format(self.curr_run))
        print('model trained and saved!')
        return


    def reset(self):
        self.load_model()
        self.load_data()
        self.curr_epoch = None
        self.curr_batch = None


    def load_data(self):
        train_dataset = MNIST(root='../data/',train=True, download=True,
            transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=self.train_batch_size, shuffle=True)
        test_dataset = MNIST(root='../data/', train=False, download=True,
            transform=transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=self.test_batch_size, shuffle=True)


    def load_model(self):
        self.model = LeNet5().to(self.device)
        # self.model = DNN2.to(self.device)
        # self.model = DNN4.to(self.device) 
        # self.model = CNN()
        # self.model = DNN4()
        self.model = self.model.to(self.device)  
        # self.adversary = LinfPGDAttack()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay)

    # (also does the testing)
    def train(self):

        # regular training
        self.model.train()

        for self.curr_epoch in range(1, self.num_epochs+1):
            train_loss = 0.
            train_correct = 0
            total = 0

            for self.curr_batch, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                # perturb data during noisy training
                if self.training_type == 'noisy':
                    perturbation = torch.randn(x.size()).to(self.device)*self.variance
                    x += perturbation
                x_var, y_var = Variable(x).to(self.device), Variable(y).to(self.device)
                outcome = self.model(x_var)
                prediction = torch.max(outcome, 1)
                train_correct += np.sum(prediction[1].cpu().numpy() == y_var.cpu().numpy())
                total += y_var.size(0)
                loss = self.criterion(outcome, y_var)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # add training on adversarial perturbation during adv training
                if self.training_type == 'adversarial':
                    x, y = self.adversary.get_adversarial_examples(
                            self.model, x, y,
                            step_size=self.adv_step_size,
                            num_examples=self.adv_num_examples,
                            max_iter=self.adv_max_iter)
                    x_var, y_var = Variable(x).to(self.device), Variable(y).to(self.device)
                    outcome = self.model(x_var)
                    prediction = torch.max(outcome, 1)
                    train_correct += np.sum(prediction[1].cpu().numpy() == y_var.cpu().numpy())
                    total += y_var.size(0)
                    loss = self.criterion(outcome, y_var)
                    train_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            train_correct, train_loss = train_correct/total, train_loss/total
            with torch.no_grad():
                # testing
                test_loss = 0.
                test_correct = 0
                total = 0
                for self.curr_batch, (x, y) in enumerate(self.test_loader):
                    #x, y = x.to(self.device), y.to(self.device)
                    x_var, y_var = Variable(x).to(self.device), Variable(y).to(self.device)
                    outcome = self.model(x_var)
                    loss = self.criterion(outcome, y_var)
                    test_loss += loss.item()
                    prediction = torch.max(outcome, 1)
                    test_correct += np.sum(prediction[1].cpu().numpy() == y_var.cpu().numpy())
                    total += y_var.size(0)
                test_correct, test_loss = test_correct / total, test_loss / total

            print('epoch {}/{}, train loss: {}, val loss: {}'.format(
                self.curr_epoch, self.num_epochs, train_loss, test_loss))

            print('train correct: {}, test correct: {}'.format(train_correct, test_correct))
      #      self.save(train_loss, train_correct.item(), test_loss, test_correct.item(), spikiness)
      #      self.write_tb(train_loss, train_correct, test_loss, test_correct, spikiness, avg_cap, avg_vol)


    def save(self, train_loss, train_class, test_loss, test_class, spikiness):
        self.save_quantities.run += [self.curr_run]
        self.save_quantities.epoch += [self.curr_epoch]
        self.save_quantities.train_loss += [train_loss]
        self.save_quantities.train_class += [train_class]
        self.save_quantities.test_loss += [test_loss]
        self.save_quantities.test_class += [test_class]
        self.save_quantities.spikiness += [spikiness]


    def write_tb(self, train_loss, train_correct, test_loss, test_correct, spikiness, avg_cap, avg_vol):
        self.writer.add_scalar('Loss/train', train_loss, self.curr_epoch)
        self.writer.add_scalar('Loss/test', test_loss, self.curr_epoch)
        self.writer.add_scalar('Accuracy/train', train_correct, self.curr_epoch)
        self.writer.add_scalar('Accuracy/test', test_correct, self.curr_epoch)


experiment = Experiment(param)
experiment.conduct()


