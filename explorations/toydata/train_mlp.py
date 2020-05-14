import torch
import sys
sys.path.append("../../")
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse

from datasets import Star
from models.toydata.models import MLPModel
import time

step = 0

def train(args, model, device, train_loader, optimizer, loss_fn, epoch, logger, exp_name):
    model.train()
    for batch_idx, data in enumerate(train_loader):

        samples, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(samples)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        global step
        step = step + 1
        if (step + 1) % 100 == 0 and args.enable_logging:
            info = {"loss_steps_" + exp_name: loss.item()}
            for tag, value in info.items():
                logger.add_scalar(tag, value, step + 1)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for samples, targets in test_loader:
            samples, targets = samples.to(device), targets.to(device)
            output = model(samples)
            test_loss += loss_fn(output, targets).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True).squeeze(dim=1) # get the index of the max log-probability
            correct += (pred == targets).sum() 
            #print(samples.shape, targets.shape, pred.shape)
            #print("Correct", correct)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
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
    parser.add_argument('--load-model', type=bool, default=False,
                        help='loads a model from model load path')
    parser.add_argument('--model-load-path', type=str,
            default='./saved_models/brockett_mlp_predictor1574095149.7499995.pt')

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

    model = MLPModel(hidden_nodes=70).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #args.load_model = False
    if args.load_model:
        checkpoint = torch.load(args.model_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    exp_name = str(time.time())

    #train loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, loss_fn, epoch,
                logger, exp_name)
        if (args.save_model):
            torch.save({
                    "model_state_dict" : model.state_dict(),
                    "optim_state_dict" : optimizer.state_dict(),
                    }, "../../data/saved_models/toydata/" + exp_name + ".pt")

            test(args, model, device, loss_fn, test_loader) 
    
if __name__=="__main__":
    main()
