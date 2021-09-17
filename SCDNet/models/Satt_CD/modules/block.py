import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import unetConv2,unetConv2_res
from models.MS_Attention.attention import ChannelAttentionHL
class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)#[2,2048,16,16]
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)#[2,2048,1,1]
        x_gpb = self.conv_gpb(x_gpb)#[2,2048,1,1]
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)#[2,512,8,8]
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)#[2,512,8,8]
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)#[2,512,4,4]
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)#[2,512,4,4]
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)#[2,512,2,2]
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)#[2,512,2,2]
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))#[2,512,4,4]
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))#[2,512,8,8]
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))#[2,2048,16,16]

        #
        out = self.relu(x_master + x_gpb)#[2,2048,16,16]+[2,2048,1,1]==>[2,2048,16,16]

        return out

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True,use_res=False):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)
        if use_res:
            self.conv_last=unetConv2_res(channels_low, channels_low, is_batchnorm=True)
        else:
            #self.conv_last = unetConv2(channels_low, channels_low, is_batchnorm=True)#lead to unetConv2.weight none when use kaiming normal
            self.conv_last=nn.Sequential(nn.Conv2d(channels_low, channels_low, 3, 1, 1,bias=False),
                                     nn.BatchNorm2d(channels_low),
                                     nn.ReLU(inplace=True))


        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)
        self.att=ChannelAttentionHL(channels_high,channels_low)
    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor. [2,2048,16,16]
        :param fms_low: Features of low level.  Tensor.  [2,1024,16,16]
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape
        #=========================================================
        # fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)#[2,2048,1,1]
        # fms_high_gp = self.conv1x1(fms_high_gp)#[2,1024,1,1]
        # # fms_high_gp = self.bn_high(fms_high_gp)
        # # fms_high_gp = self.relu(fms_high_gp)
        # fms_high_gp=F.sigmoid(fms_high_gp)
        #=====================use avg+max=========================
        fms_high_gp=self.att(fms_high)
        #=========================================================

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)#[2,1024,16,16]
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)#[2,1024,16,16]

        out=self.conv_last (out)
        return out




class GAU_CS(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True,use_res=False):
        super(GAU_CS, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)
        if use_res:
            self.conv_last=unetConv2_res(channels_low, channels_low, is_batchnorm=True)
        else:
            self.conv_last = unetConv2(channels_low, channels_low, is_batchnorm=True)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)
        self.att=ChannelAttentionHL(channels_high,channels_low)
        self.pam12_1 = nn.Sequential(
            nn.Conv2d(channels_low, channels_low // 2, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(channels_low // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_low // 2, 1, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.pam12_2 = nn.Sequential(
            nn.Conv2d(channels_low, channels_low // 2, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(channels_low // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_low // 2, 1, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor. [2,2048,16,16]
        :param fms_low: Features of low level.  Tensor.  [2,1024,16,16]
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape
        #=========================================================
        # fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)#[2,2048,1,1]
        # fms_high_gp = self.conv1x1(fms_high_gp)#[2,1024,1,1]
        # # fms_high_gp = self.bn_high(fms_high_gp)
        # # fms_high_gp = self.relu(fms_high_gp)
        # fms_high_gp=F.sigmoid(fms_high_gp)
        #=====================use avg+max=========================
        fms_high_gp=self.att(fms_high)
        #=========================================================

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)#[2,1024,16,16]
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)#[2,1024,16,16]

        att_1 = self.pam12_1(out)
        att_2 = self.pam12_2(out)
        att12 = F.sigmoid(att_1 + att_2)
        out = att12 *out

        out=self.conv_last (out)

        return out



from functools import partial
nonlinearity = partial(F.relu,inplace=True)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()
        channels_mid=int(in_channels // 4)
        self.conv1 = nn.Conv2d(in_channels, channels_mid, 1)
        self.norm1 = nn.BatchNorm2d(channels_mid)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(channels_mid, channels_mid, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(channels_mid)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(channels_mid, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

#####################################################################################################################
##########################################perpetubations decoders####################################################
#####################################################################################################################
import math
import random
import numpy as np
import cv2
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

# class torch.distributions.normal.Normal(loc, scale, validate_args=None)
# m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
'''
loc (float or Tensor) — 均值（也被称为mu）
scale (float or Tensor) — 标准差 （也被称为sigma）
'''


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """

    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels * (scale ** 2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)




class DropOutDecoder(nn.Module):
    def __init__(self, conv_in_ch, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        x = self.dropout(x)
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self,  conv_in_ch):
        super(FeatureDropDecoder, self).__init__()
       # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _):  # 其中’_’ 是一个循环标志，也可以用i，j 等其他字母代替，下面的循环中不会用到，起到的是循环此数的作用
        x = self.feature_dropout(x)
       # x = self.upsample(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self,  conv_in_ch, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)
        self.nor_dist = Normal(0, 1.0)

    def feature_based_noise(self, x):
        # noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        noise_vector = self.nor_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _):
        x = self.feature_based_noise(x)
        #x = self.upsample(x)
        return x


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax((x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = (x_detached + xi * d)  # 加入扰动的预测
        logp_hat = F.log_softmax(pred_hat, dim=1)  # 真实数据的预测
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')  # 是尽可能的在扰动上使得这两个预测最大化
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        #decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, conv_in_ch, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps  # 这里的ϵ为一个超参，控制扰动的选取的界限
        self.it = iterations
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        r_adv = get_r_adv(x, self.it, self.xi, self.eps)
        #x = self.upsample(x + r_adv)
        return x+ r_adv


def guided_cutout(output,resize,erase=0.4, use_dropout=False):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6) / 10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try:  # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:  # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w - min_w, max_h - min_h
            rnd_start_w = random.randint(0, int(bb_w * (1 - erase)))
            rnd_start_h = random.randint(0, int(bb_h * (1 - erase)))
            h_start, h_end = min_h + rnd_start_h, min_h + rnd_start_h + int(bb_h * erase)
            w_start, w_end = min_w + rnd_start_w, min_w + rnd_start_w + int(bb_w * erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class CutOutDecoder(nn.Module):
    def __init__(self, conv_in_ch, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        # self.upscale = upscale
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        maskcut = guided_cutout(pred, erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        #x = self.upsample(x)
        return x


def guided_masking(x, output, resize, return_msk_context=True):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)

    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class ContextMaskingDecoder(nn.Module):
    def __init__(self, conv_in_ch):
        super(ContextMaskingDecoder, self).__init__()
        # self.upscale = upscale
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                          return_msk_context=True)
        #x_masked_context = self.upsample(x_masked_context)
        return x_masked_context


class ObjectMaskingDecoder(nn.Module):
    def __init__(self,  conv_in_ch):
        super(ObjectMaskingDecoder, self).__init__()
        #self.upscale = upscale
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                       return_msk_context=False)
        #x_masked_obj = self.upsample(x_masked_obj)

        return x_masked_obj



#######################################################################################################################
#######################################pertubations img for mean-teacer================================================
#######################################################################################################################
class DropOutImg(nn.Module):
    def __init__(self, conv_in_ch, drop_rate=0.1, spatial_dropout=True):
        super(DropOutImg, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.dropout(x)
        return x


class FeatureDropImg(nn.Module):
    def __init__(self,  conv_in_ch):
        super(FeatureDropImg, self).__init__()
       # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        '''
        we mask 10% to 40% of the most active regions in the feature map.
        '''
    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):  # 其中’_’ 是一个循环标志，也可以用i，j 等其他字母代替，下面的循环中不会用到，起到的是循环此数的作用
        x = self.feature_dropout(x)
        return x


class FeatureNoiseImg(nn.Module):
    def __init__(self,  conv_in_ch, uniform_range=0.3,noise_type="gauss"):
        super(FeatureNoiseImg, self).__init__()
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)
        self.nor_dist = Normal(0, 1.0)
        self.noise_type=noise_type

    def feature_based_noise(self, x):
        #
        if self.noise_type=="gauss":
            noise_vector = self.nor_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        else:
            noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        #x = self.upsample(x)
        return x


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax((x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = (x_detached + xi * d)  # 加入扰动的预测
        logp_hat = F.log_softmax(pred_hat, dim=1)  # 真实数据的预测
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')  # 是尽可能的在扰动上使得这两个预测最大化
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        #decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATImg(nn.Module):
    def __init__(self, conv_in_ch, xi=1e-1, eps=10.0, iterations=1):
        super(VATImg, self).__init__()
        self.xi = xi
        self.eps = eps  # 这里的ϵ为一个超参，控制扰动的选取的界限
        self.it = iterations
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        r_adv = get_r_adv(x, self.it, self.xi, self.eps)
        #x = self.upsample(x + r_adv)
        return x+ r_adv


def guided_cutout(output,resize,erase=0.4, use_dropout=False):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6) / 10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try:  # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:  # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w - min_w, max_h - min_h
            rnd_start_w = random.randint(0, int(bb_w * (1 - erase)))
            rnd_start_h = random.randint(0, int(bb_h * (1 - erase)))
            h_start, h_end = min_h + rnd_start_h, min_h + rnd_start_h + int(bb_h * erase)
            w_start, w_end = min_w + rnd_start_w, min_w + rnd_start_w + int(bb_w * erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class CutOutImg(nn.Module):
    def __init__(self, conv_in_ch, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutImg, self).__init__()
        self.erase = erase
        # self.upscale = upscale
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        maskcut = guided_cutout(pred, erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        #x = self.upsample(x)
        return x


def guided_masking(x, output, resize, return_msk_context=True):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)

    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class ContextMaskingImg(nn.Module):
    def __init__(self, conv_in_ch):
        super(ContextMaskingImg, self).__init__()
        # self.upscale = upscale
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                          return_msk_context=True)
        #x_masked_context = self.upsample(x_masked_context)
        return x_masked_context


class ObjectMaskingImg(nn.Module):
    def __init__(self,  conv_in_ch):
        super(ObjectMaskingImg, self).__init__()
        #self.upscale = upscale
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                       return_msk_context=False)
        #x_masked_obj = self.upsample(x_masked_obj)

        return x_masked_obj












if __name__ == '__main__':
    input = torch.randn((2, 256, 8, 8))
    output = torch.randn((2, 3, 128, 128))

    upscale = 8
    num_out_ch = 2048
    decoder_in_ch = 256
    num_classes = 3
    # feature_drop = [FeatureDropDecoder(decoder_in_ch, 3)
    #                 for _ in range(2)]
    # feature_noise = [FeatureNoiseDecoder(decoder_in_ch, 3,
    #                                      uniform_range=0.3)
    #                  for _ in range(2)]
    cut_decoder = [CutOutDecoder(decoder_in_ch, erase=0.4)
                   for _ in range(1)]
    context_m_decoder = [ContextMaskingDecoder(decoder_in_ch)
                         for _ in range(1)]
    object_masking = [ObjectMaskingDecoder(decoder_in_ch)
                      for _ in range(1)]
    net = nn.ModuleList([*cut_decoder,*context_m_decoder,*object_masking])
    # print(feature_noise)
    # outputs=net[0](input,outputs0)

    outputs = [mynet(input) for mynet in net]  # 6*n module, generate 6*n seg maps of [2,3,128,128]
    # print(outputs.size())
    targets = F.softmax(output, dim=1)
    print(targets.size())
    print(outputs[0].size())
    print(outputs[1].size())


