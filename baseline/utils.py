import torch
import torch.nn.functional as F
from einops import rearrange

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

def correlation(output, labels, dim = 0):
    # assumed shape [S 15]
    # implements Pearson's Correlation
    x = output
    y = labels

    vx = x - torch.mean(x, dim=dim, keepdim=True) # mean along the temporal axis [S 15] - [1 15]
    vy = y - torch.mean(y, dim=dim, keepdim=True)

    cor = torch.sum(vx * vy, dim=dim) / (torch.sqrt(torch.sum(vx ** 2, dim=dim) * torch.sum(vy ** 2, dim=dim)) + 1e-6) # [15]
    mean_cor = torch.mean(cor)
    return mean_cor, cor

def loss_function(output, labels, criterion):
    # [B S C]
    output1 = rearrange(output, 'B S C -> (B C) S')
    labels1 = rearrange(labels, 'B S C -> (B C) S')
    t_loss = F.l1_loss(output1, labels1) # termporal loss

    output = torch.log(output)
    output2 = rearrange(output, 'B S C -> (B S) C')
    labels2 = rearrange(labels, 'B S C -> (B S) C')
    l_sum = torch.sum(labels2, dim=1)
    indices = []
    for i in range(output2.size()[0]):
        if l_sum[i] != 0.0:
            indices.append(i)

    # class loss
    loss = criterion(output2[indices], labels2[indices]) + 0.5 * t_loss # [B S 15]
    return loss