import torch
from torch import nn
from einops import rearrange

class Baseline(nn.Module):
    def __init__(self, img_feat_size=2048, au_feat_size=128, num_classes=15):
        super(Baseline, self).__init__()
        self.image_gru = nn.GRU(
            input_size=img_feat_size, hidden_size=512, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.audio_gru = nn.GRU(
            input_size=au_feat_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.contextGate1 = ContextGating(input_size=2 * (512 + 128))
        # self.fusion = nn.Linear(2 * (512+128), 2 * (512+128), bias=True)
        self.linear = nn.Linear(2 * (512+128), num_classes, bias=True)
        self.contextGate2 = ContextGating(input_size=num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, img, au):
        self.image_gru.flatten_parameters()
        self.audio_gru.flatten_parameters()
        img_gru_out = self.image_gru(img)  # [B S 512]
        au_gru_out = self.audio_gru(au)  # [B S 128]
        x = torch.cat((img_gru_out[0], au_gru_out[0]), 2) # [B S 1280]
        x = self.contextGate1(x) # [B S 1280]
        # x = self.fusion(x)
        x = self.linear(x)
        # x = torch.sigmoid(x)
        x = self.contextGate2(x) # [B S 15]
        # scale = torch.Tensor([[[10, 100, 100, 1, 10, 10, 1, 10, 100, 100, 1, 10, 1, 10, 100]]]).cuda()
        # x = x / scale
        # x = self.softmax(x)
        x = torch.sigmoid(x)
        # x = torch.clamp(x, 0.0, 1.0)
        
        return x

class EmoBase(nn.Module):
    def __init__(self, num_classes=15):
        super(EmoBase, self).__init__()
        self.image_gru = nn.GRU(
            input_size=2048, hidden_size=512, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.audio_gru = nn.GRU(
            input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.contextGate1 = ContextGating(input_size=2 * (512 + 128))
        self.linear1 = nn.Linear(2 * (512+128), num_classes, bias=True)
        self.contextGate2 = ContextGating(input_size=num_classes)
        # mean prediction
        self.contextGate3 = ContextGating(input_size=2 * (512 + 128))
        self.linear2 =  nn.Linear(2 * (512+128), num_classes, bias=True)
        self.contextGate4 = ContextGating(input_size=num_classes)
        # std prediction
        self.contextGate5 = ContextGating(input_size=2 * (512 + 128))
        self.linear3 =  nn.Linear(2 * (512+128), num_classes, bias=True)
        self.contextGate6 = ContextGating(input_size=num_classes)

    def forward(self, img, au):
        self.image_gru.flatten_parameters()
        self.audio_gru.flatten_parameters()
        img_gru_out = self.image_gru(img)  # [B S 512]
        au_gru_out = self.audio_gru(au)  # [B S 128]
        x = torch.cat((img_gru_out[0], au_gru_out[0]), 2) # [B S 1280]
        # feats = torch.cat((img, au), 2)
        # feats_mean = torch.mean(feats, dim=1) # [B 1 1280]
        feats_mean = torch.mean(x, dim=1)
        # prediction deviation
        x = self.contextGate1(x) # [B S 1280]
        x = self.linear1(x)
        x = self.contextGate2(x) # [B S 15]
        # predict mean
        x_mean = self.contextGate3(feats_mean) # [B 1280]
        x_mean = self.linear2(x_mean)
        x_mean = self.contextGate4(x_mean) # [B 15]
        # predict std
        x_std = self.contextGate5(feats_mean) # [B 1280]
        x_std = self.linear3(x_std)
        x_std = self.contextGate6(x_std) # [B 15]
        
        output = x * x_std.unsqueeze(1) + x_mean.unsqueeze(1)
        # output = torch.clamp(x, 0.0, 1.0)
        return output, x_mean, x_std

class Baseline_Img(nn.Module):
    def __init__(self, img_feat_size):
        super().__init__()
        self.image_gru = nn.GRU(
            input_size=img_feat_size, hidden_size=512, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.contextGate1 = ContextGating(input_size=2 * (512))
        self.linear = nn.Linear(2 * (512), 15, bias=True)
        self.contextGate2 = ContextGating(input_size=15)

    def forward(self, img):
        self.image_gru.flatten_parameters()
        x = self.image_gru(img)[0]  # [B S 512]
        x = self.contextGate1(x) # [B S 1024]
        x = self.linear(x)
        x = self.contextGate2(x) # [B S 15]
        # x = self.softmax(x)
        x = torch.sigmoid(x)
        return x

class Baseline_Au(nn.Module):
    def __init__(self, au_feat_size):
        super().__init__()
        self.audio_gru = nn.GRU(
            input_size=au_feat_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.contextGate1 = ContextGating(input_size=2 * (128))
        self.linear = nn.Linear(2 * (128), 15, bias=True)
        self.contextGate2 = ContextGating(input_size=15)

    def forward(self, au):
        self.audio_gru.flatten_parameters()
        x = self.audio_gru(au)[0]  # [B S 128]
        x = self.contextGate1(x) # [B S 256]
        x = self.linear(x)
        x = self.contextGate2(x) # [B S 15]
        # x = self.softmax(x)
        x = torch.sigmoid(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, input_size):
        super(ContextGating, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=True)
        # self.bn = nn.BatchNorm1d()

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
        en_out = []
        de_out = []
        for layer in self.ed_modules[0: self.layer_count]:
            x = layer(x)
            en_out.append(x)
        for layer in self.ed_modules[self.layer_count:]:
            x = layer(x)
            de_out.append(x)

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

class Encoder_TCFPN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2) # B C T time wise convolution
        self.bn = nn.BatchNorm1d(num_features=out_channels) # B C T
        self.dropout = nn.Dropout2d(0.1) # N C H W operates on C (may work with 3 dimensions)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class Mid_TCFPN(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.linear = nn.Linear(out_channels, 15)
        self.softmax = nn.Softmax(dim=1)
        self.upsample_out = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x_out = rearrange(x, 'B C T -> B T C')
        x_out = self.linear(x_out)
        x_out = rearrange(x_out, 'B T C -> B C T')
        x_out = self.softmax(x_out)
        x_out = self.upsample_out(x_out)
        return x, x_out

class Decoder_TCFPN(nn.Module):
    def __init__(self, in_channels, out_channels, aux_in, kernel_size, num_classes, scale_factor):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2) # B C T time wise convolution
        self.aux_conv = nn.Conv1d(aux_in, out_channels, 1)
        self.dropout = nn.Dropout2d(0.1) # N C H W operates on C (may work with 3 dimensions)
        self.linear = nn.Linear(out_channels, num_classes)
        self.softmax = nn.Softmax(dim=1) 
        self.upsample_out = nn.Upsample(scale_factor=scale_factor) # operates on the last dim
        self.upsample = nn.Upsample(scale_factor=2.0)

    def forward(self, x, aux):
        x = self.upsample(x)
        if aux != None:
            aux = self.aux_conv(aux)
            x = x + aux
        x = self.conv(x)
        x = self.dropout(x) # B C T
        x_out = rearrange(x, 'B C T -> B T C') # C channel
        x_out = self.linear(x_out)
        x_out = rearrange(x_out, 'B T C -> B C T')
        x_out = self.softmax(x_out) # dim 1
        x_out = self.upsample_out(x_out)
        return x, x_out

class TCFPN(nn.Module):
    def __init__(self, layers, in_channels, num_classes, kernel_size):
        super().__init__()
        self.layer_count = len(layers)
        self.layers = layers # filter count for each layer
        self.ed_modules = []
        input_size = in_channels
        # >>> Encoder Layers <<<
        for i in range(self.layer_count):
            encoder = Encoder_TCFPN(input_size, self.layers[i], kernel_size).cuda() # [B C T]
            input_size = self.layers[i]
            self.ed_modules.append(encoder)
        self.mid_layer = Mid_TCFPN(input_size, self.layers[0], 8)
        # >>> Decoder Layers <<<
        for i in range(self.layer_count - 1, -1, -1): #  2 1 0
            decoder = Decoder_TCFPN(self.layers[0], self.layers[0], self.layers[i - 1], kernel_size, num_classes, 2**i).cuda()
            input_size = self.layers[0]
            self.ed_modules.append(decoder)

    def forward(self, img, au):
        # B T C
        x = torch.cat((img, au), dim=-1)
        x = rearrange(x, 'B T C -> B C T')
        
        encode_out = []
        decode_out = []
        for mod in self.ed_modules[:len(self.ed_modules)//2]:
            x = mod(x)
            encode_out.append(x)
        x, x_out = self.mid_layer(x)
        decode_out.append(x_out)
        for i, mod in enumerate(self.ed_modules[len(self.ed_modules)//2:]):
            if i == 2:
                x, x_out = mod(x, None)
            else: # 0, 1 -> 
                x, x_out = mod(x, encode_out[1 - i])
            decode_out.append(x_out)
        x = torch.stack(decode_out, dim=0)
        avg = torch.mean(x, dim=0) # B Channel T
        out = rearrange(avg, 'B C T -> B T C')
        return out

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(torch.sigmoid(out) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]