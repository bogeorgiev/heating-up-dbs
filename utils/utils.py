import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import time
import random as rd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import eagerpy as ep


#def flatten(x: ep.Tensor, keep: int=1) => ep.Tensor:
#    return x.flatten(start=keep)
#
#
#def atleast_kd(x: ep.Tensor, k: int) -> ep.Tensor:
#    shape = x.shape + (1,) * (k-x.ndim)
#    return x.reshape(shape)


def emp_vol(dist=1.0, num_samples=10000, dim=2):
    error_rate = torch.randn(num_samples, dim + 2)
    error_rate = error_rate / error_rate.norm(dim=1).unsqueeze(1)
    error_rate = error_rate[:, :-2]
    error_rate = error_rate[:, 0]
    error_rate = (error_rate > dist).sum()
    vol = error_rate.float() / num_samples

    return vol


def planar_cap(target_vol=0.01, dim=2, precision=1.0e-3, start=0.0, end=1.0, num_samples=10000):
    """Returns hitting probability of a hyperplane given error rate of size target_vol"""

    mid = (start + end) / 2
    vol = emp_vol(dist=mid, dim=dim, num_samples=num_samples)
    while abs(vol - target_vol) > precision:
        #print("dist ", mid)
        #print("vol", vol)
        mid = (start + end) / 2
        vol = emp_vol(dist=mid, dim=dim, num_samples=num_samples)
        if vol > target_vol:
            start = mid
        else:
            end = mid
    
    dist = mid
    t = torch.tensor(1.0 / dim)
    normal_1d = Normal(torch.tensor(0.0), torch.tensor(1.0))
    cap = 2 * normal_1d.cdf(-dist / ( torch.sqrt(t)))

    if vol > 0.:
        tau = cap / vol        
    else:
        tau = 0.

    return cap, tau
    
    


def save_decision_boundary(net, dataset, boundaries, run_id, epoch, batch, device):

    # get plane with decision boundaries
    spaced_points = get_spaced_points(*boundaries).to(device)
    classification = net(spaced_points).argmax(dim=1)
    
    # write grid into file
    f = open('run_saves/boundary_run_'+run_id+'_epoch_'+str(epoch)+'_batch_'+str(batch)+'.txt', 'w')
    for i in range(len(classification)):
        f.write(
            str(spaced_points[i][0].item())+' '+
            str(spaced_points[i][1].item())+' '+
            str(classification[i].item())+
        '\n')
    
    # write training set into file for reference
    loader = DataLoader(dataset, batch_size=len(dataset))
    it = iter(loader)
    batch = next(it)
    coords, labels = batch[0], batch[1]
    for i in range(len(labels)):
        f.write(str(coords[i][0].item())+' '+str(coords[i][1].item())+' '+str(labels[i].item()+2)+'\n')
    
    f.close()
    return 


def get_distances(deltas):
    distances = []
    for delta in deltas:
        distances += [torch.sqrt(sum(sum(sum(delta**2))))]
    return torch.tensor(distances)


def get_batch(batch_size, num, *args):
    args = list(args)
    for i in range(len(args)):
        args[i] = args[i][num*batch_size:(num+1)*batch_size]
    return args


def expand(num_copies, device, *args):
    args = list(args)
    for i in range(len(args)):
        if args[i].dim() == 1:
            args[i] = torch.tensor([args[i] for j in range(num_copies)]).to(device)
        else:
            hold = [args[i] for j in range(num_copies)]
            args[i] = torch.stack(hold).to(device)
            args[i] = args[i].squeeze(1)
    return args


def get_spikiness(net, spikiness_dataset, device, adversary):
    # random walk parameters
    num_examples = 100
    batch_size = 5
    num_walks = 200
    dim = 784.
    random_walk_steps = 100
    # ratio displacement by walk and vol sampling boundaries
    factor = 1.2
    spikiness = 0.
    avg_cap = 0.
    avg_vol = 0.
    spikiness_loader = DataLoader(spikiness_dataset, batch_size=num_examples, shuffle=True)
    it = iter(spikiness_loader)
    spikiness_batch = next(it)
    x, y = spikiness_batch[0].to(device), spikiness_batch[1].to(device)

    t0 = time.time()
    dist = torch.tensor(adversary.get_distances(net, x, y, device)).to(device)
    t1 = time.time()

    for j in range(int(num_examples/batch_size)):
        x_curr, y_curr, dist_curr = get_batch(batch_size, j, x, y, dist)
       
        caps = get_caps(net, x_curr, y_curr, device, dist_curr, random_walk_steps, num_walks, dim).to(device)
        vols = get_vol(net, x_curr, y_curr, device, dist_curr*factor).to(device)
        
        # get batch wise volumes
        spikiness += sum(torch.tensor(caps).to(device)/vols).item()
        avg_cap += sum(caps).item()
        avg_vol += sum(vols).item()
    return spikiness/num_examples, avg_cap/num_examples, avg_vol/num_examples


def get_caps(net, x, y, device, step_sizes, num_steps, num_walks):
    caps = []
    for i in range(len(y)):
        curr_example = x[i:(i+1)]
        curr_label = y[i:(i+1)]
        curr_step = step_sizes[i:(i+1)]
        caps += [get_one_cap(net, curr_example, curr_label, device, curr_step, num_steps, num_walks, i)]
    return caps
    

def get_one_cap(net, x, y, device, step, num_steps, num_walks, j):
    walks, labels = expand(num_walks, device, x, y)
    killed_walks = torch.ones(num_walks).to(device)
    
    for i in range(num_steps):
        walks = walks + torch.randn(walks.size(), device=device) * step
        outcome = net(walks)
        _, pred = torch.max(outcome.data, 1)
        killed_walks[pred != labels] *= 0.
        if killed_walks.sum() == 0.:
            break
    cap = (num_walks - killed_walks.sum()) / num_walks
    #print('Example ', j, ' Capacity ', cap.item())
    return cap.item()


def get_vols(net, x, y, device, radius, num_samples):
    vols = []
    for i in range(len(y)):
        curr_example = x[i:(i+1)]
        curr_label = y[i:(i+1)]
        curr_radius = radius[i]
        vols += [get_one_vol(net, curr_example, curr_label, device, curr_radius, num_samples)]
    return vols


# new volume method (sampling unformly from ball)
def get_one_vol(net, x, y, device, radius, num_samples, sample_full_ball=False, shape=[3, 32, 32]):
    """
        x: 1 x dim 
        y: 1 x num_labels
        shape: dataset sample shape (CIFAR10 default)
    """
    dim = 1
    for axis in shape:
        dim *= axis

    if sample_full_ball: 
        sample_step = torch.randn(num_samples, dim + 2).to(device)
        sample_step = radius * sample_step / sample_step.norm(dim=1).unsqueeze(1)
        sample_step = sample_step[:, :-2]
    else:
        sample_step = radius * torch.randn(num_samples, dim).to(device)

    shape = [num_samples] + shape
    sample_step = sample_step.view(shape)
    #print(sample_step)
    
    _, pred = torch.max(net(x + sample_step), 1)
    #print(pred)
    correct = pred.eq(y).sum()
    #print("CORRECT", correct)
    vol = (num_samples - correct.item()) / num_samples

    return vol


def get_spaced_points(x_min, x_max, y_min, y_max, steps):
    points = torch.zeros(steps*steps, 2)
    for i in range(steps):
        for j in range(steps):
            points[i*steps + j] = torch.tensor(
                [x_min+i*(x_max-x_min)/(steps-1), y_min+j*(y_max-y_min)/(steps-1)])
    return points
    
    
def plot_points(data):
    color_pallet = ['g', 'k']
    label_zero_x = []
    label_zero_y = []
    label_one_x = []
    label_one_y = []
    
    for idx in range(len(data)):
        if data[idx].label == 0:
            label_zero_x += [data[idx].x.item()]
            label_zero_y += [data[idx].y.item()]
        else:
            label_one_x += [data[idx].x.item()]
            label_one_y += [data[idx].y.item()]
      
    plt.scatter(np.array(label_zero_x), np.array(label_zero_y), c=color_pallet[0], s=2)
    plt.scatter(np.array(label_one_x), np.array(label_one_y), c=color_pallet[1], s=2)
    plt.axis('equal')
    return


def save_results_to_file(result, filename):
    f = open('run_saves/'+filename, 'w')
    for i in range(len(result)):
        f.write(str(result[i].spikiness.item()) + ' ' + str(result[i].loss.item()) + 
                ' ' + str(result[i].epoch) + ' ' + str(result[i].batch)+ '\n')
    f.close()
    return


def get_validation_error(net, val_set):
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    it = iter(val_loader)
    val_batch = next(it)
    coordinates, labels = val_batch[0], val_batch[1]
    classification = net(coordinates)
    _, classification = classification.max(-1)
    missclassification = sum(abs((classification-labels).numpy()))/len(val_set)
    return round(missclassification*100, 1)


class Savestate:
    def __init__(self, filename):
        self.filename = filename
        # current run of the experiment
        self.run = []
        self.epoch = []
        self.train_loss = []
        # current correct classification during training
        self.train_class = []
        # current correct classification during testing
        self.test_class = []
        self.test_loss = []
        self.spikiness = []

    def write_in_file(self):
        f = open(self.filename, 'w')
        for i in range(len(self.run)):
            f.write(
                    str(self.run[i])+' '+
                    str(self.epoch[i])+' '+
                    str(self.train_loss[i])+' '+
                    str(self.train_class[i])+' '+
                    str(self.test_loss[i])+' '+
                    str(self.test_class[i])+' '+
                    str(self.spikiness[i])+'\n'
                    )
        f.close()
