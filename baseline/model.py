import torch
from torch import nn
from einops import rearrange

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
        # x = self.softmax(x)
        x = torch.sigmoid(x)
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


class ED_TCN(nn.Module):
    def __init__(self, layers, in_channels, num_classes, kernel_size):
        super(ED_TCN, self).__init__()
        self.layer_count = len(layers)
        self.layers = layers # filter count for each layer
        self.ed_modules = []
        input_size = in_channels
        # >>> Encoder Layers <<<
        for i in range(self.layer_count):
            encoder = Encoder_TCN(input_size, self.layers[i], kernel_size) # [B C T]
            input_size = self.layers[i]
            self.ed_modules.append(encoder)
        # >>> Decoder Layers <<<
        for i in range(self.layer_count - 1, -1, -1):
            decoder = Decoder_TCN(input_size, self.layers[i], kernel_size)
            input_size = self.layers[i]
            self.ed_modules.append(decoder)
        self.linear = nn.Linear(input_size, num_classes) # [B T C]


    def forward(self, x):
        # x: [B FeatDim Duration]
        # Conv1d works on termporal dimension
        # filter count controls channel sizes
        for layer in self.ed_modules:
            x = layer(x)
        x = rearrange(x, 'B C T -> B T C')
        x = self.linear(x) # B T 15
        x = torch.sigmoid(x)
        return x


class Encoder_TCN(nn.Module):
    # an encoder layer in ED-TCN
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Encoder_TCN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout2d(0.3) # N C H W operates on C (may work with 3 dimensions)
        self.activation = Norm_Relu(dim=1) # opeartes on frames
        self.pool = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        # x: [B FeatDim Duration]
        x = self.conv(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class Decoder_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Decoder_TCN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout2d(0.3)
        self.activation = Norm_Relu(dim=1)
    
    def forward(self,  x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.activition(x)
        return x


class Norm_Relu(nn.Module):
    # operate on frames
    def __init__(self, dim=1):
        super(Norm_Relu, self).__init__()
        self.dim = dim # axis to apply normalization

    def forward(self, x):
        x = nn.functional.relu(x)
        mx = torch.max(x, dim=self.dim, keepdim=True)
        return x / (mx + 1e-5)

