import torch
from torch import nn
from einops import rearrange
from torch.nn import init
######################################################################
class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net" 
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * (1-ratio))
        self.BN = nn.BatchNorm1d(self.half)
        self.IN = nn.InstanceNorm1d(planes - self.half, affine=True)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.BN(split[0].contiguous())
        out2 = self.IN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.2, relu=False, bnorm=True, num_bottleneck=256, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        #self.bn= nn.BatchNorm1d(num_bottleneck)
        self.bn= IBN(num_bottleneck)
        classifier = []
        if droprate>0:
            classifier += [nn.Dropout(p=droprate)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = x.transpose(1,-1)
        x = self.bn(x)
        x = x.transpose(1,-1)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x



class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.image_gru = nn.GRU(
            input_size=2048, hidden_size=512, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.audio_gru = nn.GRU(
            input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.contextGate_i1 = ContextGating(input_size=2 * 512)
        self.contextGate_a1 = ContextGating(input_size=2 * 128)
        self.image_linear = ClassBlock(2 * 512, 15)
        self.audio_linear = ClassBlock(2 * 128, 15) 
        self.contextGate_i2 = ContextGating(input_size=15)
        self.contextGate_a2 = ContextGating(input_size=15)
        self.softmax = nn.Softmax(dim=2)
        self.p = nn.Parameter(torch.ones(())*0, requires_grad = True)

    def forward(self, img, au):
        self.image_gru.flatten_parameters()
        self.audio_gru.flatten_parameters()
        img_gru_out = self.image_gru(img)  # [B S 512]
        au_gru_out = self.audio_gru(au)  # [B S 128]
        #x = torch.cat((img_gru_out[0], au_gru_out[0]), 2) # [B S 1280]
        xi = self.contextGate_i1(img_gru_out[0]) # [B S 1280]
        xi = self.image_linear(xi)
        xi = self.contextGate_i2(xi) # [B S 15]
        # x = self.softmax(x)
        xa = self.contextGate_a1(au_gru_out[0]) # [B S 1280]
        xa = self.audio_linear(xa)
        xa = self.contextGate_a2(xa) # [B S 15]
        p = torch.sigmoid(self.p)
        x = torch.sigmoid(p*xi+(1-p)*xa)
        xi = torch.sigmoid(xi)
        xa = torch.sigmoid(xa)
        return x, xi, xa


class ContextGating(nn.Module):
    def __init__(self, input_size):
        super(ContextGating, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=True)
        self.linear.apply(weights_init_kaiming)

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
