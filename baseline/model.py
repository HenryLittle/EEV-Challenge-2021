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
        return x


class ContextGating(nn.Module):
    def __init__(self, input_size):
        super(ContextGating, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.size())
        wx = self.linear(x)
        # print(wx.size())
        gates = self.sigmoid(wx)
        return gates * x



class Correlation(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
