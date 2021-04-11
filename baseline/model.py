import torch
from torch import nn

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.image_gru = nn.GRU(
            input_size=2048, hidden_size=512, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.audio_gru = nn.GRU(
            input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.contextGate1 = ContextGating(input_size=2 * (512 + 128))
        self.linear = nn.Linear(2 * (512+128), 15, bias=True)
        self.contextGate2 = ContextGating(input_size=15)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, img, au):
        self.image_gru.flatten_parameters()
        self.audio_gru.flatten_parameters()
        img_gru_out = self.image_gru(img)  # [B S 512]
        au_gru_out = self.audio_gru(au)  # [B S 128]
        x = torch.cat((img_gru_out[0], au_gru_out[0]), 2) # [B S 1280]
        x = self.contextGate1(x) # [B S 1280]
        x = self.linear(x)
        x = self.contextGate2(x) # [B S 15]
#        x = self.softmax(x)
        x = torch.sigmoid(x) # force to [0, 1]
        return x


class ContextGating(nn.Module):
    def __init__(self, input_size):
        super(ContextGating, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        # print(x.size())
        wx = self.linear(x)
        # print(wx.size())
        gates = torch.sigmoid(wx)
        return gates * x



class Correlation(nn.Module):
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, x, y):
        vx = x - torch.mean(x, dim=0, keepdim=True) # mean along the temporal axis [S 15] - [1 15]
        vy = y - torch.mean(y, dim=0, keepdim=True)

        cor = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0) * torch.sum(vy ** 2, dim=0)) + 1e-6) # [15]
        mean_cor = torch.mean(cor)
        return 1 - mean_cor
