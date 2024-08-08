
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchaudio
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from torchvision.models.resnet import resnet18, resnet101, resnet34
import torch.nn.functional as F
import math
import copy
from einops import rearrange


class CrossLearning(nn.Module):
    def __init__(self, opt={}):
        super(CrossLearning, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']
        self.cross_attention1 = Cross_Attention(dim=self.embed_dim, heads=8, dim_head=32)
        self.cross_attention2 = Cross_Attention(dim=self.embed_dim, heads=8, dim_head=32)

    def forward(self, x, y):

        batch_x = x.size(0)
        batch_y = y.size(0)
        x_1 = x.unsqueeze(dim=1).expand(-1, batch_y, -1)
        y_1 = y.unsqueeze(dim=0).expand(batch_x, -1, -1)
        x_2 = x.unsqueeze(dim=0).expand(batch_y, -1, -1)
        y_2 = y.unsqueeze(dim=1).expand(-1, batch_x, -1)

        x_cross = self.cross_attention1(y_1, x_1)
        y_cross = self.cross_attention2(x_2, y_2).transpose(0, 1)

        x_final = x_1 + torch.sigmoid(y_cross) * x_cross + torch.sigmoid(y_1) * x_cross
        y_final = y_1 + torch.sigmoid(y_cross) * y_1 + torch.sigmoid(x_cross) * y_1

        return x_final, y_final


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class ExtractFeature(nn.Module):
    def __init__(self, opt={}, finetune=True):
        super(ExtractFeature, self).__init__()

        self.resnet = resnet18(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = finetune


        self.f0conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.f01conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=4, padding=3)

        self.f2conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.fusionconv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.quater_att_fusion = QUATER_ATTENTION(512)
        self.quater_att_f4 = QUATER_ATTENTION(512)

    def forward(self, img):

        x = self.resnet.conv1(img)  # [30, 64, 128, 128]
        x = self.resnet.bn1(x)
        f0 = self.resnet.relu(x)
        x = self.resnet.maxpool(f0)  # [30, 64, 64, 64]

        f1 = self.resnet.layer1(x)  # [30, 64, 64, 64]
        f2 = self.resnet.layer2(f1)  # [30, 128, 32, 32]
        # x = self.ca_att(x)
        f3 = self.resnet.layer3(f2)  # [30, 256, 16, 16]
        f4 = self.resnet.layer4(f3)  # [30, 512, 8, 8]
        # x = self.ca_att(x)

        # multi scale module
        f0 = self.f0conv(f0)
        f_0_1 = torch.cat([f0, f1], dim=1)
        f_0_1 = self.f01conv(f_0_1)

        f2 = self.f2conv(f2)
        # f3 = self.ca_att_f3(f3)
        f_2_3 = torch.cat([f2, f3], dim=1)
        # f_2_3 = self.ca_att_f23(f_2_3)

        f_fusion = torch.cat([f_0_1, f_2_3], dim=1)
        f_fusion = self.fusionconv(f_fusion)
        f_fusion = self.quater_att_fusion(f_fusion)
        f_fusion_score = torch.sigmoid(f_fusion)

        f4 = self.quater_att_f4(f4)
        f4_score = torch.sigmoid(f4)

        final = f4 * f_fusion_score + f4_score * f_fusion  # [30, 512, 8, 8]

        final += f4

        x = self.resnet.avgpool(final)   # [30, 512, 1, 1]
        x = torch.flatten(x, 1)      # [30, 512]

        return x


class QUATER_ATTENTION(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(QUATER_ATTENTION, self).__init__()

        self.fc_h = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn_h = nn.BatchNorm2d(in_planes // ratio)
        self.relu_h = nn.ReLU()
        self.conv_h_sptial = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio, 7, padding=3, bias=False)

        self.fc_w = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn_w = nn.BatchNorm2d(in_planes // ratio)
        self.relu_w = nn.ReLU()
        self.conv_w_sptial = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio, 7, padding=3, bias=False)

        self.fc_general = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h_avg = torch.mean(x, dim=3, keepdim=True)
        x_h_max, _ = torch.max(x, dim=3, keepdim=True)
        x_h_max = x_h_max

        x_w_avg = torch.mean(x, dim=2, keepdim=True)
        x_w_max, _ = torch.max(x, dim=2, keepdim=True)

        x_h_avg = self.relu_h(self.bn_h(self.fc_h(x_h_avg)))
        x_h_max = self.relu_h(self.bn_h(self.fc_h(x_h_max)))

        x_w_avg = self.relu_w(self.bn_w(self.fc_w(x_w_avg)))
        x_w_max = self.relu_w(self.bn_w(self.fc_w(x_w_max)))

        x_h_cat_sp = self.conv_h_sptial(torch.cat([x_h_avg, x_h_max], dim=1))
        x_w_cat_sp = self.conv_w_sptial(torch.cat([x_w_avg, x_w_max], dim=1))

        x_h_w = x_h_cat_sp * x_w_cat_sp

        x_general = self.fc_general(x_h_w)
        res = x * self.sigmoid(x_general)

        return res




class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class VoiceFeature(nn.Module):
    def __init__(self, opt = {}):
        super(VoiceFeature, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']

        self.torchfbank = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.conv1 = nn.Conv1d(80, 1024, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        C = 1024

        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, self.embed_dim)
        self.bn6 = nn.BatchNorm1d(self.embed_dim)

        self.ca_att = CA_Block(1536)

    def forward(self, x):
        with torch.no_grad():
            # print('x_shape_input:', x.shape)

            x = self.torchfbank(x)+1e-6

            # print('x_shape_fbank:', x.shape)

            x = x.log()

            # print('x_shape_log:', x.shape)

            x = x - torch.mean(x, dim=-1, keepdim=True)

            # print('x_shape_mean:', x.shape)

        x = self.conv1(x)
        # print('x_shape_conv1:', x.shape)
        x = self.relu(x)
        # print('x_shape_relu:', x.shape)
        x = self.bn1(x)
        # print('x_shape_bn1:', x.shape)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        #ca_att
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
        x = self.ca_att(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))  # torch.Size([16, 1536, 404])

        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)  # torch.Size([16, 3072])
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)  # torch.Size([16, 512])

        # print('x_shape_final-->:', x.shape)

        return x


def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

