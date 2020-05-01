import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from utils import *


class Adversary:
    def __init__(self, strategy, device, eps=None, alpha=None, num_iter=None, restarts=None):
        self.strategy = strategy
        self.eps = eps
        self.alpha = alpha
        self.num_iter = num_iter
        self.restarts = restarts
        self.criterion = nn.CrossEntropyLoss()
        self.device = device


    def get_adversarial_examples(self, model, x, y):
        if self.strategy == 'fgsm':
            return self.fgsm(model, x, y, self.eps or 0.1)

        elif self.strategy == 'pgd':
            return self.pgd(model, x, y, self.eps or 0.1,
                    self.alpha or 1e4, self.num_iter or 1000)

        elif self.strategy == 'pgd_linf':
            return self.pgd_linf(model, x, y, self.eps or 0.1,
                    self.alpha or 1e-2, self.num_iter or 40)
            
        elif self.strategy == 'pgd_linf_rand':
            return self.pgd_linf_rand(model, x, y, self.eps or 0.1,
                    self.alpha or 1e-2, self.num_iter or 40, self.restarts or 10)

    def fgsm(self, model, x, y, eps):    
        x, y = x.to(self.device), y.to(self.device)
        delta = torch.zeros_like(x, requires_grad=True, device=self.device)
        output = model(x+delta)
        loss = self.criterion(output, y)
        loss.backward()
        return eps * delta.grad.detach().sign()
     

    def pgd(self, model, x, y, eps, alpha, num_iter):
        x, y = x.to(self.device), y.to(self.device)
        delta = torch.zeros_like(x, requires_grad=True).to(x.device)
        for t in range(num_iter):
            loss = self.criterion(model(x + delta), y)
            loss.backward()
            delta.data = (delta + x.shape[0]*alpha*delta.grad.data).clamp(-eps,eps)
            delta.grad.zero_()
        return delta.detach()
        

    def pgd_linf(self, model, x, y, eps, alpha, num_iter):
        x, y = x.to(self.device), y.to(self.device)
        delta = torch.zeros_like(x, requires_grad=True).to(x.device)
        for t in range(num_iter):
            loss = self.criterion(model(x + delta), y)
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-eps,eps)
            delta.grad.zero_()
        return delta.detach()


    def pgd_linf_rand(self, model, x, y, eps, alpha, num_iter, restarts):
        x, y = x.to(self.device), y.to(self.device)
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = torch.zeros_like(x)

        for i in range(restarts):
            delta = torch.rand_like(x, requires_grad=True).to(x.device)
            delta.data = delta.data * 2 * eps - eps

            for t in range(num_iter):
                loss = self.criterion(model(x + delta), y)
                loss.backward()
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-eps,eps)
                delta.grad.zero_()

            all_loss = nn.CrossEntropyLoss(reduction='none')(model(x+delta), y)
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
        return max_delta


    # return array with l_2 distances to closest adv examples of x
    def get_distances(self, model, x, y, device, eps=0.1, alpha=1.0e-2, max_iter=1000):
        # travel in adversarial direction until adversary is found
        # using pgd_linf method
        self.device = device
        # 0: not yet found, 1: found
        tracker = torch.zeros_like(y).to(device)
        distances = torch.zeros(y.size()).to(device).float()
        delta = torch.zeros_like(x, requires_grad=True, device=self.device)
        org_x = x 

        for i in range(max_iter):
            x.requires_grad = True
            model.zero_grad()
            loss = self.criterion(model(x), y).to(device)
            loss.backward()
            adv_x = x + alpha * x.grad.sign() 
            eta = torch.clamp(adv_x - org_x, min=-eps, max=eps)
            x = (org_x + eta).detach_()

            # eval current predictions
            _, pred = torch.max(model(x).data, 1)
            killed = (pred != y and tracker == 0)
            tracker[killed] = 1
            distances[killed] = torch.norm((x - org_x).view(len(y), -1), dim=1)[killed]

            if sum(tracker) == len(tracker):
                break

        return distances, tracker


    def perturb(self, data, device, variance):
        return data + torch.randn(data.size()).to(device)*np.sqrt(variance)



class FlatClassifier(nn.Module):
    def __init__(self, dim=2):
        super(FlatClassifier, self).__init__()
        self.dim = dim
        self.l1 = nn.Linear(self.dim, 2, bias=False)
        self.l1.weight.data = torch.zeros(2, self.dim, requires_grad=True)
        self.l1.weight.data[0,0] = 1.
        self.l1.weight.data[1,0] = -1.
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.l1(x))



if __name__ == "__main__":
    f = FlatClassifier()
    x = torch.tensor([10.3, 0.]).unsqueeze(0)
    y = torch.tensor([0])
    adv = Adversary(strategy="fgsm", device=torch.device('cpu'))
    print(f(x))
    print('distance: ', adv.get_distances(model=f, x=x, y=y, device=torch.device('cpu')))



