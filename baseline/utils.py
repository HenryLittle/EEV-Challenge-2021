import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def correlation(output, labels, dim = 1):
    # assumed shape [S 15]
    # implements Pearson's Correlation
    x = output
    y = labels

    vx = x - torch.mean(x, dim=dim, keepdim=True) # mean along the temporal axis [S 15] - [1 15]
    vy = y - torch.mean(y, dim=dim, keepdim=True)

    cor = torch.sum(vx * vy, dim=dim) / (torch.sqrt(torch.sum(vx ** 2, dim=dim) * torch.sum(vy ** 2, dim=dim)) + 1e-6) # [15]
    mean_cor = torch.mean(cor)
    return mean_cor, cor