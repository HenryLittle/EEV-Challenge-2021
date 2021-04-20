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

def interpolate_output(output, in_freq, out_freq):
    # output [Time Cls]
    scale = int(out_freq // in_freq)
    length = output.size()[0] # time length
    out_length = scale * (length - 1) + 1 # make sure each sample point is aligned
    output = F.interpolate(rearrange(output, '(1 T) C -> 1 C T'), out_length, mode='linear', align_corners=True)
    output = rearrange(output, '1 C T -> (1 T) C')
    # print(length, out_length, output.size()[0])
    return output

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

def loss_function(output, labels, validate=False):
    # [B S C]
    # output1 = rearrange(output, 'B S C -> (B C) S')
    # labels1 = rearrange(labels, 'B S C -> (B C) S')
    if validate:
        t_loss = F.l1_loss(output, labels)
        loss = t_loss
    else:
        output1 = rearrange(output, 'B S C -> (B C) S')
        labels1 = rearrange(labels, 'B S C -> (B C) S')
        t_loss = F.l1_loss(output1,  labels1)

        output2 = rearrange(output, 'B S C -> (B S) C')
        labels2 = rearrange(labels, 'B S C -> (B S) C')
        c_loss = F.l1_loss(output2, labels2)
        # c_loss = F.kl_div(torch.log(output2), labels2)
        loss = t_loss + c_loss

    return loss