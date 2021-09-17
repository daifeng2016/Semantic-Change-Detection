
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
from torch import nn

#torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'semanticModule']#表示一个模块中允许哪些属性可以被导入到别的模块中



class _EncoderBlock(Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class semanticModule(Module):
    """ Semantic attention module"""
    def __init__(self, in_dim):
        super(semanticModule, self).__init__()
        self.chanel_in = in_dim

        self.enc1 = _EncoderBlock(in_dim, in_dim*2)
        self.enc2 = _EncoderBlock(in_dim*2, in_dim*4)
        self.dec2 = _DecoderBlock(in_dim * 4, in_dim * 2, in_dim * 2)
        self.dec1 = _DecoderBlock(in_dim * 2, in_dim, in_dim )

    def forward(self,x):

        enc1 = self.enc1(x)#[1,64,128,128]==>[1,128,64,64]
        enc2 = self.enc2(enc1)#[1,128,64,64]==>[1,256,32,32]

        dec2 = self.dec2( enc2)#[1,256,32,32]==>[1,128,64,64]
        dec1 = self.dec1( F.upsample(dec2, enc1.size()[2:], mode='bilinear'))#[1,128,64,64]==>[1,64,128,128]

        return enc2.view(-1), dec1#

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))#define a trainable parameter

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()#[1,64,128,128]
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)#[1,64,128,128]==>[1,8,128,128]==>[1,8,16384]==>[1,16384,8]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)#[1,64,128,128]==>[1,8,128,128]==>[1,8,16384]

        energy = torch.bmm(proj_query, proj_key)#[1,16384,8].[1,8,16384]==>[1,16384,16384]
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)##[1,64,128,128]==>[1,64,128,128]==>[1,64,16384]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))#[1,64,16384].[1,16384,16384]==>[1,64,16384]
        out = out.view(m_batchsize, C, height, width)#[1,64,16384]==>[1,64,128,128]

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))#不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.softmax  = Softmax(dim=-1)#对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1 ,dim:指明维度，dim=0表示按列计算；dim=1表示按行计算。默认dim的方法已经弃用了
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()#[1,3,128,128]
        proj_query = x.view(m_batchsize, C, -1)#[1,3,16384]
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)#[1,16384,3]
       
        energy = torch.bmm(proj_query, proj_key)#[1,3,16384].[1,16384,3]=[1,3,3]
        #torch.max()[0]， 只返回最大值的每个数  troch.max()[1]， 只返回最大值的每个索引

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy#[1,3,3]
        attention = self.softmax(energy_new)#[1,3,3]
        proj_value = x.view(m_batchsize, C, -1)##[1,3,16384]

        out = torch.bmm(attention, proj_value)#[1,3,3].[1,3,16384]=[1,3,16384]  == weights = torch.matmul(affinity_new, proj_value)
        out = out.view(m_batchsize, C, height, width)#[1,3,16384]==>[1,64,128,128]

        out = self.gamma*out + x
        return out
'''
https://blog.csdn.net/ruoruojiaojiao/article/details/89074763
我们将s设置为比例尺寸的控制参数，也就是可以将输入通道数平均等分为多个特征通道。
s越大表明多尺度能力越强，此外一些额外的计算开销也可以忽略
'''

class Res2NetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,scale=4,expansion=1,use_se=False):#the output channels is out_channels*expansion
        super(Res2NetBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.scale=scale
        self.expansion=expansion
        self.use_se=use_se
        split_channels=out_channels//scale

        self.conv1s_bn=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3_bn = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(split_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1e_bn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*expansion, 1),
            nn.BatchNorm2d(out_channels*expansion),
            nn.ReLU(inplace=True)
        )
        self.conv1_bn_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * expansion, 1),
            nn.BatchNorm2d(out_channels * expansion),
            nn.ReLU(inplace=True)
        )

        se_channels = self.out_channels * self.expansion
        neck_channels = int(se_channels // 16)
        self.conv_se1=nn.Sequential(
            nn.Conv2d(se_channels,neck_channels,1),
            nn.ReLU(inplace=True)
        )
        self.conv_se2=nn.Sequential(
            nn.Conv2d(neck_channels,se_channels,1),
            nn.Sigmoid()
        )

    def SE_Block(self,x):

        se_branch0=F.adaptive_avg_pool2d(x,(1,1))
        se_branch1=self.conv_se1(se_branch0)
        se_branch2=self.conv_se2(se_branch1)

        return torch.mul(x,se_branch2)# equals to x*se_branch2 mul==dot product mm=matrixmultipy

    def forward(self, x):
        input_x=x
        x=self.conv1s_bn(x)
        x_splits=[]
        split_channels=int(self.out_channels//self.scale)
        for i in range(self.scale):
            x_slice=x[:,i*split_channels:(i+1)*split_channels,...]
            if i>1:
                x_slice=torch.add(x_slice,x_splits[-1])
            if i>0:
                x_slice=self.conv3_bn(x_slice)
            x_splits.append(x_slice)
        x_splits_con=torch.cat((x_splits[0],x_splits[1],x_splits[2],x_splits[3]),dim=1)
        x_conv=self.conv1e_bn(x_splits)
        if self.use_se:
            x_conv=self.SE_Block(x_conv)

        x_skip=self.conv1_bn_skip(input_x)
        return x_conv+x_skip
#================================================
#===============SCAM=============================
'''
reference:Landslide detection from an open satellite imagery
and digital elevation model dataset using attention
boosted convolutional neural networks
'''
class SCAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)#16 is too large for remote sensing images?
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))#[1,24,128,128]==>[1,24,1,1]==>[1,24,1,1]
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))#
        # out = avg_out + max_out
        ca=self.avg_pool(x)
        cm=self.max_pool(x)
        sa = torch.mean(x, dim=1, keepdim=True)
        sm, _ = torch.max(x, dim=1, keepdim=True)
        sca=ca*sa
        scm=cm*sm
        avg_out = self.fc2(self.relu1(self.fc1(sca)))
        max_out = self.fc2(self.relu1(self.fc1(scm)))
        out = avg_out + max_out
        return self.sigmoid(out)
#================================================
#==============CBAM Attention====================
'''
 self.ca = ChannelAttention(planes)
 self.sa = SpatialAttention()
 out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel/reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = x * chn_se

        spa_se = self.spatial_se(x)
        spa_se = x * spa_se
        return chn_se + spa_se
#
# def channel_attention(self, num_channel, ablation=False):
#     # todo add convolution here
#     pool = nn.AdaptiveAvgPool2d(1)
#     conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
#     # bn = nn.BatchNorm2d(num_channel)
#     activation = nn.Sigmoid()  # todo modify the activation function
#
#     return nn.Sequential(*[pool, conv, activation])

class ChannelAttention1(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv=nn.Conv2d(in_planes, in_planes, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out=self.avg_pool(x)
        conv_out=self.conv(avg_out)

        return self.sigmoid(conv_out)




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)#16 is too large for remote sensing images?
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))#[1,24,128,128]==>[1,24,1,1]==>[1,24,1,1]
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))#
        avg_out=self.fc(self.avg_pool(x))#seem to work better SE-like attention
        max_out=self.fc(self.max_pool(x))
        out = avg_out + max_out
        #out = avg_out
        return self.sigmoid(out)


class ChannelAttentionHL(nn.Module):
    def __init__(self, high_ch,low_ch):
        super(ChannelAttentionHL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Conv2d(high_ch, low_ch, 1, bias=False)
        # self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)#16 is too large for remote sensing images?
        # self.relu1 = nn.ReLU()
        # self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))#[1,24,128,128]==>[1,24,1,1]==>[1,24,1,1]
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))#
        avg_out=self.fc(self.avg_pool(x))#seem to work better SE-like attention
        max_out=self.fc(self.max_pool(x))
        out = avg_out + max_out
        #out = avg_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#=============================BAM attenion E:\TEST2020\DownLoadPrj\CD\TGS-SaltNet-master\TGS-SaltNet-master\bam.py
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        #self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM_Att(nn.Module):
    def __init__(self, gate_channel):
        super(BAM_Att, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + torch.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor




if __name__ == '__main__':
    xs = torch.randn(size=(1, 32, 128, 128))
    # sa_model = semanticModule(64)  # init only input model config
    # output = sa_model(xs)  # forward input tensor
    ch_att=ChannelAttention(32)
    sp_att=SpatialAttention()
    output_ch=ch_att(xs)*xs#[1,32,1,1]*[1,32,128,128]=[1,32,128,128]
    output_sa=sp_att(xs)*xs#[1,1,128,128]*[1,32,128,128]=[1,32,128,128]
    print(output_ch.size())
    print(output_sa.size())