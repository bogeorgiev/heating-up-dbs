import math

dataset = 'cifar10'
num_runs = 1
num_epochs = 200
lr = 0.1
train_batch_size = 128
test_batch_size = 100
weight_decay = 5e-4

# available training types 'normal', 'noisy', 'adversarial'
training_type = 'adversarial'
# available strategies 'fgsm', 'pgd', 'pgd_linf', 'pgd_linf_rand'
attack = 'fgsm'
# for noisy training
variance = 0.1

save_file_quants = dataset+'_'+training_type+'_save.txt'
if training_type == 'noisy':
    save_file_model = training_type+'_'+str(variance)+'_WideResNet_28_10_run_'
elif training_type == 'adversarial':    
    save_file_model = training_type+'_'+attack+'_WideResNet_28_10_run_'
else:
    save_file_model = training_type+'_WideResNet_28_10_run_'

save_path_model = './cifar_model_saves/'

mean = {'cifar10': (0.4914, 0.4822, 0.4465)}
std = {'cifar10': (0.2023, 0.1994, 0.2010)}

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    optim_factor = 0
    if (epoch > 160):
        optim_factor = 3
    elif (epoch > 120):
        optim_factor = 2
    elif (epoch > 60):
        optim_factor = 1
    return init * math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s





