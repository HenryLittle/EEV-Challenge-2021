import torch
import torch.nn.functional as F
from einops import rearrange
import pickle as pkl

f = open('/data0/EEV/code/tools/mean_std.pkl', 'rb')
data_meanstd = pkl.load(f)
mean_all = torch.from_numpy(data_meanstd['mean']).cuda()
std_all = torch.from_numpy(data_meanstd['std']).cuda()

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
    # assumed shape [B S 15]
    # implements Pearson's Correlation
    # x = output * std_all.unsqueeze(0) + mean_all.unsqueeze(0)
    x = output
    y = labels

    vx = x - torch.mean(x, dim=dim, keepdim=True) # mean along the temporal axis [B S 15] - [B 1 15]
    vy = y - torch.mean(y, dim=dim, keepdim=True)

    cor = torch.sum(vx * vy, dim=dim) / (torch.sqrt(torch.sum(vx ** 2, dim=dim) * torch.sum(vy ** 2, dim=dim)) + 1e-6) # [B 15]
    mean_cor = torch.mean(cor)
    return mean_cor, cor # 1, [B 15]

def ccc(x, y, dim=1):
    # B T C
    # Concordance Cor Coe
    mean_x = torch.mean(x, dim=dim, keepdim=True)
    mean_y = torch.mean(y, dim=dim, keepdim=True)
    vx = x - mean_x # mean along the temporal axis [B T C] - [B 1 C]
    vy = y - mean_y
    cov = 2 * torch.sum(vx*vy, dim=dim) # B C
    x_var = torch.var(x, dim=dim) # B C
    y_var = torch.var(y, dim=dim)
    denom = x_var + y_var + (mean_x - mean_y)**2
    ccc = cov / denom # B C
    return 1.0 - ccc 

def mean_constraint(output, labels):
    output1 = rearrange(output, 'B (S S1) C -> B S S1 C', S1=5)
    output1 = torch.mean(output1, dim=2) # B S' C
    labels1 = rearrange(labels, 'B (S S1) C -> B S S1 C', S1=5)
    labels1 = torch.mean(labels1, dim=2)
    loss = F.l1_loss(output1, labels1)
    return loss


def loss_function(output, labels, args, mean=None, std=None, validate=False):
    # [B S C]
    # output1 = rearrange(output, 'B S C -> (B C) S')
    # labels1 = rearrange(labels, 'B S C -> (B C) S')
    if validate:
        # (S C)
        # labels = (labels - mean_all.unsqueeze(0)) / std_all.unsqueeze(0)
        c_loss = F.l1_loss(output, labels)
        t_loss = F.l1_loss(rearrange(output, 'S C -> C S'), rearrange(labels, 'S C -> C S'))
        loss = t_loss + c_loss
    else:
        # labels = (labels - mean_all.unsqueeze(0).unsqueeze(0)) / std_all.unsqueeze(0).unsqueeze(0)
        
        # c_loss = mean_constraint(output, labels)
        # c_loss = ccc(output, labels, dim=1)
        # c_loss = 0.1 * torch.mean(c_loss)
        if args.cls_mask != None:
            # mask = [1 if x in args.cls_mask else 0 for x in range(0, 15)]
            t_loss = F.l1_loss(output[:, :, args.cls_mask], labels[:, :, args.cls_mask]) # B S C
        else:
            t_loss = F.l1_loss(output, labels) # B S C
            # t_loss = ccc(output, labels, dim=1)
            # t_loss = 0.2 * torch.mean(t_loss)


        # t_loss = F.l1_loss(output, labels, reduction='none') # B S C
        # t_loss = torch.mean(t_loss, dim=1) # B C 
        # _, pc_corr = correlation(output, labels, dim=1) # [B S C] -> [B C]
        # weights = 1.0 - pc_corr
        # # weights = F.softmax(weights, dim=1) # B C
        # # # print(weights.size(), weights)
        # t_loss = torch.sum(t_loss*weights, dim=1)
        # t_loss = torch.mean(t_loss)
        # breakpoint()
        if mean != None and std != None:
            l_mean = torch.mean(labels, dim=1) # B C
            m_loss = F.mse_loss(mean, l_mean)
            l_std = torch.std(labels, dim=1)
            s_loss = F.mse_loss(std, l_std)

            c_loss = m_loss + s_loss
        else:
            c_loss = F.l1_loss(output, labels) # B S C

        # c_loss = F.kl_div(torch.log(output2), labels2)
        loss = t_loss + c_loss

    return loss, c_loss, t_loss

def loss_functionC(output, labels, validate=False):
    # [B S C]
    # output1 = rearrange(output, 'B S C -> (B C) S')
    # labels1 = rearrange(labels, 'B S C -> (B C) S')
    if validate:
        t_loss = F.l1_loss(output, labels)
        loss = t_loss
    else:
        s = torch.sum(torch.sum(labels, dim=-1), dim=-1)
        s[s>0] = 1
        s = s.unsqueeze(-1).unsqueeze(-1)
        output = output * s.expand_as(output)
        output1 = rearrange(output, 'B S C -> (B C) S')
        labels1 = rearrange(labels, 'B S C -> (B C) S')
        t_loss = F.l1_loss(output1,  labels1)

        #output2 = rearrange(output, 'B S C -> (B S) C')
        #labels2 = rearrange(labels, 'B S C -> (B S) C')
        #c_loss = F.l1_loss(output2, labels2)
        # c_loss = F.kl_div(torch.log(output2), labels2)
        #loss = t_loss + c_loss

    return t_loss

def loss_function_zzd(pred1, pred2, labels, validate=False):
    # [B S C]
    # output1 = rearrange(output, 'B S C -> (B C) S')
    # labels1 = rearrange(labels, 'B S C -> (B C) S')
    if validate:
        t_loss = F.l1_loss(pred1, labels)
        loss = t_loss
    else:
        output1 = rearrange(pred1, 'B S C -> (B C) S')
        labels1 = rearrange(labels, 'B S C -> (B C) S')
        t_loss = F.l1_loss(output1,  labels1)

        output2 = rearrange(pred1, 'B S C -> (B S) C')
        labels2 = rearrange(labels, 'B S C -> (B S) C')
        c_loss = F.l1_loss(output2, labels2)

        output1 = rearrange(pred2, 'B S C -> (B C) S')
        labels1 = rearrange(labels, 'B S C -> (B C) S')
        t_loss += F.l1_loss(output1,  labels1)

        output2 = rearrange(pred2, 'B S C -> (B S) C')
        labels2 = rearrange(labels, 'B S C -> (B S) C')
        c_loss += F.l1_loss(output2, labels2)

        # c_loss = F.kl_div(torch.log(output2), labels2)
        loss = t_loss + c_loss 

        #variance = torch.sum(F.l1_loss(pred1),self.sm(pred2)), dim=1) 
        #exp_variance = torch.exp(-variance)
        #loss = torch.mean(loss*exp_variance) + torch.mean(variance)

    return loss

