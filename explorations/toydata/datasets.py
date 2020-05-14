import torch
import math
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class Star(Dataset):
    """Starshaped dataset"""
    def __init__(self, line_pts=50, lines=25, margin=5.):
        super(Dataset, self).__init__()
        self.size = line_pts * lines
        self.data = torch.tensor([])

        for line in range(lines):
            angle = torch.tensor(line * math.pi / lines)
            t = (2 * torch.randint(0, 2, [line_pts]) - 1).float()
            ps = torch.rand(line_pts)
            xs = torch.cos(angle) * t * (ps + margin)
            ys = torch.sin(angle) * t * (ps + margin)
            label = 0
            if line % 2 == 1:
                label = 1
            self.data = torch.cat([self.data, 
                torch.stack([xs, ys, label * torch.ones(line_pts)]).t()])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return [self.data[idx, :2], self.data[idx, 2].long()]

if __name__=="__main__":
    ds = Star()
    print(ds.data.shape)
    print(len(ds))
    print(ds[238])
    for sample in ds:
        color = "red"
        if sample[1] == 1.:
            color = "blue"
        plt.scatter(sample[0][0], sample[0][1], color=color)

    plt.show()
