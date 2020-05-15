import torch
import sys
sys.path.append("../../")
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse

from datasets import Star
from models.toydata.models import MLPModel
import time
import numpy as np

from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='MLP model')

    parser.add_argument('--batch-size', type=int, default=1250, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--dataset-size', type=int, default=200,
                        help='size of train dataset')
    parser.add_argument('--epochs', type=int, default=300000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--enable-logging', type=bool, default=False,
                        help='enables results plotting and tensorboard')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='enable model saving')
    parser.add_argument('--model-load-path', type=str,
            default='../../data/saved_models/toydata/mlp_100hu_5l.pt')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    star_ds = Star()

    logger = SummaryWriter("./logs/")
    
    train_loader = torch.utils.data.DataLoader(star_ds,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(star_ds,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = MLPModel(hidden_nodes=100).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint = torch.load(args.model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    print("Model loaded")
    model.eval()

    int_length = 1.
    start = -15
    end = 15
    num_xs = 1000
    num_ys = 1000
    xs = torch.linspace(start, end, num_xs)
    ys = torch.linspace(start, end, num_ys)

    for x in xs:
        inp = ys.unsqueeze(1).float().to(device)
        inp = torch.cat([inp, x.data*torch.ones_like(inp)], dim=1)
        out = model(inp)
        pred = out.argmax(dim=1)
        plt.scatter(inp[:, 0].cpu(), inp[:, 1].cpu(), c=pred.cpu())
    
    plt.show()

if __name__=="__main__":
    main()
