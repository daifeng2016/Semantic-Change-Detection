import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import models.Satt_CD.modules.architecture as arch
from models.MS_Attention.attention import SCSEBlock,BAM_Att
from dropblock import DropBlock2D, LinearScheduler
from models.utils import unetConv2,unetConv2_res
import logging
logger = logging.getLogger('base')
from .architecture import Dblock,ASPP,DenseASPP,FPAv2
from .block import ChannelAttentionHL
from .pyconv import PyConv3

class F_mynet3(nn.Module):
    def __init__(self, backbone='resnet18',in_c=3, f_c=64, output_stride=8,drop_rate=0.5):
        self.in_c = in_c
        super(F_mynet3, self).__init__()
        self.module = mynet3(backbone=backbone, output_stride=output_stride, f_c=f_c, in_c=self.in_c,drop_rate=drop_rate)
    def forward(self, input):
        return self.module(input)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def ResNet34(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 64
    """
    print(in_c)
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])
    return model

def ResNet18(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 256, 128, 64, 64
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, in_c=in_c)
    if in_c !=3:
        pretrained=False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet18'])
    return model

def ResNet50(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    2048, 256
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c !=3:
        pretrained=False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet50'])
    return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = BatchNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,  block, layers, output_stride, BatchNorm, pretrained=True, in_c=3):

        self.inplanes = 64
        self.in_c = in_c
        print('in_c: ',self.in_c)
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]# no enlarge  ROF
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 4:
            strides = [1, 1, 1, 1]
            dilations = [1, 2, 4, 8]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))#blocks=[1,2,4]
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)#[2,3,256,256]==>[2,64,128,128]    resnet34: [2,3,256,256]==>[2,64,128,128]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # | 4  [2,64,64,64]     resnet34:==>[2,64,64,64]
        x = self.layer1(x)  # | 4   [2,256,64,64]     resnet34:==>[2,64,64,64]
        low_level_feat2 = x  # | 4
        x = self.layer2(x)  # | 8   [2,512,32,32]    resnet34:==>[2,128,32,32]
        low_level_feat3 = x
        x = self.layer3(x)  # | 16  [2,1024,16,16]   resnet34:==>[2,256,16,16]
        low_level_feat4 = x
        x = self.layer4(x)  # | 32  #[2,2048,16,16]  resnet34:==>[2,512,16,16]
        return x, low_level_feat2, low_level_feat3, low_level_feat4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)



def build_backbone(backbone, output_stride, BatchNorm, in_c=3):
    if backbone == 'resnet50':
        return ResNet50(output_stride, BatchNorm, in_c=in_c)
    elif backbone == 'resnet34':
        return ResNet34(output_stride, BatchNorm, in_c=in_c)
    elif backbone == 'resnet18':
        return ResNet18(output_stride, BatchNorm, in_c=in_c)
    else:
        raise NotImplementedError



class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, fc,backbone, BatchNorm,drop_rate):
        super(Decoder, self).__init__()
        self.fc = fc
        #self.drop_rate=drop_rate
        if backbone=='resnet50':
           filters = [256, 512, 1024, 2048]
        else:
           filters=[64,128,256,512]
        self.dr2 = DR(filters[0], 96)
        self.dr3 = DR(filters[1], 96)
        self.dr4 = DR(filters[2], 96)
        self.dr5 = DR(filters[3], 96)
        self.last_conv = nn.Sequential(
                                       #nn.Dropout(0.5),
                                       nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(drop_rate),
                                       nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU()
                                       )

        self._init_weight()
        #self.drop_out = nn.Dropout(0.5)


    def forward(self, x,low_level_feat2, low_level_feat3, low_level_feat4):
        #[2,2048,16,16]  [2,256,64,64] [2,512,32,32] [2,1024,16,16]
        # x1 = self.dr1(low_level_feat1)
        x2 = self.dr2(low_level_feat2)#[16,64,32,32]==>[16,96,32,32]
        x3 = self.dr3(low_level_feat3)#[16,128,16,16]==>[16,96,16,16]
        x4 = self.dr4(low_level_feat4)#[16,256,8,8]==>[16,96,8,8]
        x = self.dr5(x)#[16,512,4,4]==>[16,96,4,4]
        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)#[16,96,32,32]
        # x2 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)#[16,96,32,32]
        x4 = F.interpolate(x4, size=x2.size()[2:], mode='bilinear', align_corners=True)#[16,96,32,32]

        x = torch.cat((x, x2, x3, x4), dim=1)#[16,384,32,32]

        #x=self.drop_out(x)

        x = self.last_conv(x)#[16,64,32,32]

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(fc, backbone, BatchNorm,drop_rate):
    return Decoder(fc,backbone, BatchNorm,drop_rate)


class mynet3(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=16, f_c=64, freeze_bn=False, in_c=3,drop_rate=0.2):
        super(mynet3, self).__init__()
        print('arch: mynet3')
        BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)
        self.decoder = build_decoder(f_c, backbone, BatchNorm,drop_rate)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, f2, f3, f4 = self.backbone(input)#[16,512,4,4] [16,64,32,32] [16,128,16,16] [16,256,8,8]
        x = self.decoder(x, f2, f3, f4)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

##########################################################################################################
############################################CD Model######################################################
##########################################################################################################
from .block_CD import BAM,PAM,PCAM
class FeatCmp_Net(nn.Module):

    def __init__(self,in_c=3, use_att=False,nf=64,
                 att_mode='BAM',backbone='resnet50',drop_rate=0.2):
        super(FeatCmp_Net, self).__init__()



        self.feat_Extactor=F_mynet3(backbone=backbone, in_c=in_c,f_c=nf, output_stride=16,drop_rate=drop_rate)

        if use_att:
            if att_mode=='BAM':
                self.AttFunc=BAM(nf, ds=1)
            else:
                self.AttFunc=PAM(in_channels=nf, out_channels=nf, sizes=[1,2,4,8],ds=1)
        #self.drop_out=nn.Dropout(p=0.2)

    def forward(self, x1,x2):

        feat1=self.feat_Extactor(x1)
        feat2=self.feat_Extactor(x2)

        # height = x1.shape[3]
        # # not in c-dim, thus output can be separated, 2-D conv can be used
        # x = torch.cat((x1, x2), 3)  # 2[1,64,16,16]==>[1,64,16,32]
        # x = self.Self_Att(x)
        # return x[:, :, :, 0:height], x[:, :, :, height:]

        height = feat1.shape[3]
        feat12 = torch.cat((feat1, feat2), 3)  # 2[1,64,16,16]==>[1,64,16,32]
        feat12= self.AttFunc(feat12)
        return feat12[:, :, :, 0:height], feat12[:, :, :, height:]


        #return self.AttFunc(feat1),self.AttFunc(feat2)

class EDLoc_Net(nn.Module):

    def __init__(self,in_c=3, use_att=False,nf=64,
                 att_mode='BAM'):
        super(EDLoc_Net, self).__init__()



        self.feat_Extactor=F_mynet3(backbone='resnet34', in_c=in_c,f_c=nf, output_stride=16)

        if use_att:
            if att_mode=='BAM':
                self.AttFunc=BAM(nf, ds=1)
            else:
                self.AttFunc=PAM(in_channels=nf, out_channels=nf, sizes=[1,2,4,8],ds=1)

    def forward(self, x1,x2):

        feat1=self.feat_Extactor(x1)
        feat2=self.feat_Extactor(x2)
        return self.AttFunc(feat1),self.AttFunc(feat2)


class Classifier_Module(nn.Module):

    def __init__(self, nf,dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(nf, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

#===using label32===========
class EDCls_Net(nn.Module):

    def __init__(self, in_c=3, use_att=False, nf=64,out_c=32,
                 att_mode='BAM',backbone='resnet34'):
        super(EDCls_Net, self).__init__()

        self.feat_Extactor = F_mynet3(backbone=backbone, in_c=in_c, f_c=nf, output_stride=16)

        if use_att:
            logger.info("using att for center block...")
            if att_mode == 'BAM':
                self.AttFunc = BAM(nf, ds=1)
            else:
                self.AttFunc = PAM(in_channels=nf, out_channels=nf, sizes=[1, 2, 4, 8], ds=1)


        self.classifier=self._make_pred_layer(Classifier_Module,nf*2, [1, 2, 4, 8], [1, 2, 4, 8],out_c)


    def _make_pred_layer(self, block,nf, dilation_series, padding_series, num_classes):
            return block(nf,dilation_series, padding_series, num_classes)
    def forward(self, x1, x2):

        feat1 = self.feat_Extactor(x1)#[8,64,64,64]
        feat2 = self.feat_Extactor(x2)
        feat1=self.AttFunc(feat1)
        feat2=self.AttFunc(feat2)
        feat12=torch.cat([feat1,feat2],dim=1)
        pred=self.classifier(feat12)

        pred=F.interpolate(pred,size=x1.shape[2:], mode='bilinear', align_corners=True)

        return pred

#======using label7 directly======================
class EDCls_Net2(nn.Module):

    def __init__(self, in_c=3, use_att=False, nf=64,out_c=7,
                 att_mode='BAM',backbone='resnet34',drop_rate=0.5):
        super(EDCls_Net2, self).__init__()

        self.feat_Extactor = F_mynet3(backbone=backbone, in_c=in_c, f_c=nf, output_stride=16)
        self.use_att=use_att
        if use_att:
            if att_mode == 'BAM':
                self.AttFunc = BAM(nf, ds=1)
            else:
                self.AttFunc = PAM(in_channels=nf, out_channels=nf, sizes=[1, 2, 4, 8], ds=1)


        self.classifier=self._make_pred_layer(Classifier_Module,nf*2, [1, 2, 4, 8], [1, 2, 4, 8],out_c)
        #self.drop_out=nn.Dropout(drop_rate)#p=0.5 is too large

    def _make_pred_layer(self, block,nf, dilation_series, padding_series, num_classes):
            return block(nf,dilation_series, padding_series, num_classes)
    def forward(self, x1, x2):

        feat1 = self.feat_Extactor(x1)#[8,64,64,64]
        feat2 = self.feat_Extactor(x2)
        #=======methods1 using PAM for feat1, feat2
        # feat1=self.AttFunc(feat1)
        # feat2=self.AttFunc(feat2)
        #======methods2 using PAM for feat12======
        height = feat1.shape[3]
        feat12 = torch.cat((feat1, feat2), 3)  # 2[1,64,16,16]==>[1,64,16,32]
        feat12 = self.AttFunc(feat12)
        feat1,feat2=feat12[:, :, :, 0:height], feat12[:, :, :, height:]
        #================================

        feat_diff=F.relu(feat1-feat2)#must use relu

        #feat_diff = (feat1 - feat2)#lead to bad performance

        pred1=self.classifier(torch.cat([feat1,feat_diff],dim=1))
        pred2=self.classifier(torch.cat([feat2,feat_diff],dim=1))


        pred1=F.interpolate(pred1,size=x1.shape[2:], mode='bilinear', align_corners=True)
        pred2 = F.interpolate(pred2, size=x1.shape[2:], mode='bilinear', align_corners=True)

        return pred1,pred2



class EDCls_UNet(nn.Module):

   def __init__(self, in_c=3, use_att=False, nf=64,out_c=7,net_feat=None):
        super(EDCls_UNet, self).__init__()
        self.feat_Extactor=net_feat
        self.classifier=nn.Conv2d(32*2,out_c,1,stride=1, padding=0)


   def forward(self, x1, x2):
        feat1=self.feat_Extactor(x1)
        feat2=self.feat_Extactor(x2)

        feat_diff = F.relu(feat1 - feat2)  # must use relu

        pred1 = self.classifier(torch.cat([feat1, feat_diff], dim=1))
        pred2 = self.classifier(torch.cat([feat2, feat_diff], dim=1))

        return  pred1,pred2


class EDCls_UNet2_MC7Bin(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.1,se_block='BAM'):
        super(EDCls_UNet2_MC7Bin, self).__init__()
        self.feat_Extactor=net_feat
        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.training_mode=training_mode


        if use_att:#=====can be removed

            deconv_att=True
            logger.info("use att for decoder...")
        else:
            deconv_att=False
            logger.info("no att for decoder...")

        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0,
                stop_value=drop_rate,
                nr_steps=10000
            )
        # logger.info("use residual for decoder...")
        # self.decoder4 = unetUp3_Res(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se)
        # self.decoder3 = unetUp3_Res(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se)
        # self.decoder2 = unetUp3_Res(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se)
        # self.decoder1 = unetUp3_Res(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se)
        #self.decoder4 = unetUp2(filters[4], filters[3], use_att=deconv_att, act_mode=act_mode)
        #==========================================================
        logger.info("use new residual for decoder...")
        self.decoder4 = unetUp3(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
        self.decoder3 = unetUp3(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
        self.decoder2 = unetUp3(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
        self.decoder1 = unetUp3(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
        self.decoder4_diff = unetUp2(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode)
        self.decoder3_diff = unetUp2(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode)
        self.decoder2_diff = unetUp2(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode)
        self.decoder1_diff = unetUp2(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode)

        self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1)
                                        )

        self.class_conv_bin = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, 1, 3, padding=1),
                                        nn.Sigmoid()
                                        )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )

                self.class_conv1_bin = nn.Sequential(
                    nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                    conv_act,
                    nn.Conv2d(32, 32, 3, padding=1),
                    conv_act,
                    self.dropblock,
                    nn.Conv2d(32, 1, 3, padding=1),
                    nn.Sigmoid()
                )
                self.class_conv2_bin = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                 )
                self.class_conv3_bin = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                 )


            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))

                self.classifier1_bin = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid())
                self.classifier2_bin = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                     )
                self.classifier3_bin = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid())
        else:
            logger.info("using no deep supervsion...")



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))

        d4_1 = self.decoder4(feat12_4, feat12_3, feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1, feat12_2, feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1, feat12_1, feat1_1)  # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1, feat12_0, feat1_0)  # 128

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128

        d4_12 = self.decoder4_diff(feat12_4, feat12_3)  # 16
        d3_12 = self.decoder3_diff(d4_12, feat12_2)  # 32
        d2_12 = self.decoder2_diff(d3_12, feat12_1)  # 64
        d1_12 = self.decoder1_diff(d2_12, feat12_0)  # 128



        pred1 = self.class_conv(d1_1)
        pred2 = self.class_conv(d1_2)
        pred12=self.class_conv_bin(d1_12)
        if self.training_mode:
            if self.use_DS:

                pred1_0, pred2_0,pred12_0 = self.class_conv1(d4_1), self.class_conv1(d4_2),self.class_conv1_bin(d4_12)
                pred1_1, pred2_1,pred12_1 = self.class_conv2(d3_1), self.class_conv2(d3_2),self.class_conv2_bin(d3_12)
                pred1_2, pred2_2,pred12_2 = self.class_conv3(d2_1), self.class_conv3(d2_2),self.class_conv3_bin(d2_12)

                return (pred1_0, pred2_0, pred12_0), (pred1_1, pred2_1,pred12_1), (pred1_2, pred2_2, pred12_2), (pred1, pred2, pred12)

        return  pred1,pred2, pred12


class EDCls_UNet2_MC6Bin(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.1,se_block='BAM'):
        super(EDCls_UNet2_MC6Bin, self).__init__()
        self.feat_Extactor=net_feat
        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.training_mode=training_mode


        if use_att:#=====can be removed

            deconv_att=True
            logger.info("use att for decoder...")
        else:
            deconv_att=False
            logger.info("no att for decoder...")

        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0,
                stop_value=drop_rate,
                nr_steps=10000
            )

        #==========================================================
        logger.info("use new residual for decoder...")
        # self.decoder4 = unetUp3(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
        #                                 use_se=use_se)
        # self.decoder3 = unetUp3(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
        #                                 use_se=use_se)
        # self.decoder2 = unetUp3(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
        #                                 use_se=use_se)
        # self.decoder1 = unetUp3(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
                                        #use_se=use_se)
        self.decoder4= unetUp2(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode)
        self.decoder3= unetUp2(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode)
        self.decoder2= unetUp2(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode)
        self.decoder1= unetUp2(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode)

        self.decoder4_diff = unetUp2(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode)
        self.decoder3_diff = unetUp2(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode)
        self.decoder2_diff = unetUp2(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode)
        self.decoder1_diff = unetUp2(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode)

        self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1)
                                        )

        self.class_conv_bin = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, 1, 3, padding=1),
                                        nn.Sigmoid()
                                        )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )

                self.class_conv1_bin = nn.Sequential(
                    nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                    conv_act,
                    nn.Conv2d(32, 32, 3, padding=1),
                    conv_act,
                    self.dropblock,
                    nn.Conv2d(32, 1, 3, padding=1),
                    nn.Sigmoid()
                )
                self.class_conv2_bin = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                 )
                self.class_conv3_bin = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                 )


            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))

                self.classifier1_bin = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid())
                self.classifier2_bin = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                     )
                self.classifier3_bin = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid())
        else:
            logger.info("using no deep supervsion...")



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))

        d4_1 = self.decoder4(feat1_4,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1, feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1, feat1_1)  # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1, feat1_0)  # 128

        d4_2 = self.decoder4(feat2_4, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2,  feat2_2)  # 32
        d2_2 = self.decoder2(d3_2,feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat2_0)  # 128

        d4_12 = self.decoder4_diff(feat12_4, feat12_3)  # 16
        d3_12 = self.decoder3_diff(d4_12, feat12_2)  # 32
        d2_12 = self.decoder2_diff(d3_12, feat12_1)  # 64
        d1_12 = self.decoder1_diff(d2_12, feat12_0)  # 128



        pred1 = self.class_conv(d1_1)
        pred2 = self.class_conv(d1_2)
        pred12=self.class_conv_bin(d1_12)
        if self.training_mode:
            if self.use_DS:

                pred1_0, pred2_0,pred12_0 = self.class_conv1(d4_1), self.class_conv1(d4_2),self.class_conv1_bin(d4_12)
                pred1_1, pred2_1,pred12_1 = self.class_conv2(d3_1), self.class_conv2(d3_2),self.class_conv2_bin(d3_12)
                pred1_2, pred2_2,pred12_2 = self.class_conv3(d2_1), self.class_conv3(d2_2),self.class_conv3_bin(d2_12)

                return (pred1_0, pred2_0, pred12_0), (pred1_1, pred2_1,pred12_1), (pred1_2, pred2_2, pred12_2), (pred1, pred2, pred12)

        return  pred1,pred2, pred12



class unetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu',use_se=False,se_block='SCSE',drop_block=None):
        super(unetUp3, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)
        if use_se:
            if se_block=='SCSE':
                self.conv = nn.Sequential(
                    # drop_block,#use drop after cat
                    nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    conv_act,
                    nn.Conv2d(out_size, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    conv_act,
                    SCSEBlock(out_size)  # seem to worse on the test
                )
            else:
                self.conv = nn.Sequential(
                    # drop_block,#use drop after cat
                    nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    conv_act,
                    nn.Conv2d(out_size, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    conv_act,
                    BAM_Att(out_size)  # seem to worse on the test
                )

        else:
            self.conv = nn.Sequential(nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
                                      nn.BatchNorm2d(out_size),
                                      conv_act,
                                      nn.Conv2d(out_size, out_size, 3, 1, 1),
                                      nn.BatchNorm2d(out_size),
                                      conv_act
                                      )




        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(in_size, out_size)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2


    def forward(self, inputs1, inputs2,inputs3):
        outputs1 = self.up(inputs1)
        if self.use_att:
            #=========methods1======
            att_map=self.att(inputs1)
            inputs2=att_map*inputs2
            inputs3=att_map*inputs3
            #===========method2==================
            # outputs123=torch.cat([outputs1,inputs2,inputs3], 1)
            # att_map=self.att(outputs123)
            # inputs2=inputs2*att_map
            # inputs3=inputs3*att_map


        return self.conv(torch.cat([outputs1,inputs2,inputs3], 1))

class unetUp3_Py(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu',use_se=False):
        super(unetUp3_Py, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)
        if use_se:
            self.conv = nn.Sequential(
                                      #nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
                                      PyConv3(out_size * 3, out_size),#seem to work worse
                                      nn.BatchNorm2d(out_size),
                                      conv_act,
                                      # nn.Conv2d(out_size, out_size, 3, 1, 1),
                                      # nn.BatchNorm2d(out_size),
                                      # conv_act
                                      SCSEBlock(out_size)
                                      )
        else:
            self.conv = nn.Sequential(PyConv3(out_size * 3, out_size),
                                      nn.BatchNorm2d(out_size),
                                      conv_act,
                                      PyConv3(out_size, out_size),
                                      nn.BatchNorm2d(out_size),
                                      conv_act
                                      )




        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(in_size, out_size)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2


    def forward(self, inputs1, inputs2,inputs3):
        outputs1 = self.up(inputs1)
        if self.use_att:
            #=========methods1======
            att_map=self.att(inputs1)
            inputs2=att_map*inputs2
            inputs3=att_map*inputs3
            #===========method2==================
            # outputs123=torch.cat([outputs1,inputs2,inputs3], 1)
            # att_map=self.att(outputs123)
            # inputs2=inputs2*att_map
            # inputs3=inputs3*att_map


        return self.conv(torch.cat([outputs1,inputs2,inputs3], 1))


class ResBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        ):

        super(ResBlock, self).__init__()

        self.conv_main = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),#not use?
                                      )
        self.conv_short=nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch))
    def forward(self, x):
        x_main=self.conv_main(x)
        x_short=self.conv_short(x)

        return x_main+x_short

class ResBlock2(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        ):

        super(ResBlock2, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),
                                      # nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                      # nn.BatchNorm2d(out_ch),
                                      # nn.ReLU(inplace=True),
                                      )
        self.conv2=nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch))
    def forward(self, x):
        x0=self.conv1(x)
        x_out=self.conv2(x0)

        return x0+x_out


class unetUp3_Res(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu',use_se=False):
        super(unetUp3_Res, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)
        # if use_se:
        #     self.conv = nn.Sequential(nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               conv_act,
        #                               # nn.Conv2d(out_size, out_size, 3, 1, 1),
        #                               # nn.BatchNorm2d(out_size),
        #                               # conv_act
        #                               #SCSEBlock(out_size)
        #                               )
        # else:
        #     self.conv = nn.Sequential(nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               conv_act,
        #                               nn.Conv2d(out_size, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               conv_act
        #                               )
        self.SEBlock=SCSEBlock(out_size)
        # self.conv_cat=nn.Sequential(nn.Conv2d(out_size*3, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               conv_act)
        # self.conv2=nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               #conv_act
        #                          )
        self.conv_cat=ResBlock(out_size*3,out_size)


        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(in_size, out_size)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2


    def forward(self, inputs1, inputs2,inputs3):
        outputs1 = self.up(inputs1)
        if self.use_att:
            #=========methods1======
            att_map=self.att(inputs1)
            inputs2=att_map*inputs2
            inputs3=att_map*inputs3

        conv_out=self.conv_cat(torch.cat([outputs1,inputs2,inputs3], 1))



        return conv_out


class unetUp3_50(nn.Module):
    def __init__(self, up_in, x_in, n_out, is_deconv=True,use_att=False,act_mode='relu',use_se=False,drop_block=None):
        super(unetUp3_50, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)
        if drop_block:
            self.drop_block=drop_block
        if use_se:
            self.conv = nn.Sequential(

                                      nn.Conv2d(n_out + x_in * 2, n_out, 3, 1, 1),
                                      nn.BatchNorm2d(n_out),
                                      conv_act,

                                      # nn.Conv2d(out_size, out_size, 3, 1, 1),
                                      # nn.BatchNorm2d(out_size),
                                      # conv_act
                                      SCSEBlock(n_out)
                                      )
        else:
            self.conv = nn.Sequential(nn.Conv2d(n_out + x_in * 2, n_out, 3, 1, 1),
                                      nn.BatchNorm2d(n_out),
                                      conv_act,

                                      # nn.Conv2d(out_size, out_size, 3, 1, 1),
                                      # nn.BatchNorm2d(out_size),
                                      # conv_act

                                      )





        if is_deconv:
            self.up = nn.ConvTranspose2d(up_in, n_out, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(up_in, n_out)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2


    def forward(self, inputs1, inputs2=None,inputs3=None):
        outputs1 = self.up(inputs1)
        if self.use_att:
            #=========methods1======
            att_map=self.att(inputs1)
            inputs2=att_map*inputs2
            inputs3=att_map*inputs3

        if inputs2:
            return self.conv(self.drop_block(torch.cat([outputs1, inputs2, inputs3]), 1))
        else:
            return self.conv(outputs1)

class unetUp3_64(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu',use_se=False):
        super(unetUp3_64, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)
        self.inter_conv=nn.Conv2d(out_size, 64, 3, 1, 1)#self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        if use_se:
            self.conv = nn.Sequential(nn.Conv2d(64*3, 64, 3, 1, 1),
                                      nn.BatchNorm2d(64),
                                      conv_act,
                                      # nn.Conv2d(out_size, out_size, 3, 1, 1),
                                      # nn.BatchNorm2d(out_size),
                                      # conv_act
                                      SCSEBlock(64)
                                      )
        else:
            self.conv = nn.Sequential(nn.Conv2d(64*3, 64, 3, 1, 1),
                                      nn.BatchNorm2d(64),
                                      conv_act,
                                      nn.Conv2d(64, 64, 3, 1, 1),
                                      nn.BatchNorm2d(64),
                                      conv_act
                                      )




        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, 64, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(64, 64)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2


    def forward(self, inputs1, inputs2,inputs3):
        outputs1 = self.up(inputs1)
        outputs2=self.inter_conv(inputs2)
        outputs3=self.inter_conv(inputs3)
        if self.use_att:
            #=========methods1======
            att_map=self.att(outputs1)
            outputs2=att_map*outputs2
            outputs3=att_map*outputs3



        return self.conv(torch.cat([outputs1,outputs2,outputs3], 1))


class unetUp1_64(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu',use_se=False):
        super(unetUp1_64, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)
        self.inter_conv=nn.Conv2d(out_size, 64, 3, 1, 1)
        if use_se:
            self.conv = nn.Sequential(nn.Conv2d(out_size,out_size , 3, 1, 1),
                                      nn.BatchNorm2d(out_size),
                                      conv_act,
                                      # nn.Conv2d(out_size, out_size, 3, 1, 1),
                                      # nn.BatchNorm2d(out_size),
                                      # conv_act
                                      SCSEBlock(out_size)
                                      )
        else:
            self.conv = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                      nn.BatchNorm2d(32),
                                      conv_act,
                                      nn.Conv2d(32, 64, 3, 1, 1),
                                      nn.BatchNorm2d(64),
                                      conv_act
                                      )




        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)




    def forward(self, inputs1):
        outputs1 = self.up(inputs1)
        return self.conv(outputs1)








class unetUp2(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu'):
        super(unetUp2, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(in_size, out_size)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2

        self.conv_cat = ResBlock(out_size * 2, out_size)

    def forward(self, inputs1, inputs2):
        outputs1 = self.up(inputs1)
        if self.use_att:
            #=========methods1======
            att_map=self.att(inputs1)
            inputs2=att_map*inputs2#diff
            # inputs3=att_map*inputs3#feat1
            # inputs4 = att_map * inputs4#feat2
            #===========method2==================
            # outputs123=torch.cat([outputs1,inputs2,inputs3], 1)
            # att_map=self.att(outputs123)
            # inputs2=inputs2*att_map
            # inputs3=inputs3*att_map


        return self.conv_cat(torch.cat([outputs1,inputs2], 1))


class unetUp2_Flow(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False,use_att=False,act_mode='relu'):
        super(unetUp2_Flow, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        # if is_deconv:
        #     self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        # else:
        #     self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        #
        # self.use_att=use_att
        # if self.use_att:
        #     self.att = ChannelAttentionHL(in_size, out_size)#methods1
        #     #self.att=ChannelAttentionHL(out_size*3,out_size)#method2
        self.down_h = nn.Conv2d(in_size, out_size, 1, bias = False)
        self.down_l = nn.Conv2d(out_size, out_size, 1, bias=False)
        self.flow_make = nn.Conv2d(out_size * 2, 2, kernel_size=3, padding=1, bias=False)

        self.conv_cat = ResBlock(out_size, out_size)

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, inputs1, inputs2):
        # outputs1 = self.up(inputs1)
        # if self.use_att:
        #     #=========methods1======
        #     att_map=self.att(inputs1)
        #     inputs2=att_map*inputs2#diff
        #return self.conv_cat(torch.cat([outputs1, inputs2], 1))
        h_feature, low_feature = inputs1,inputs2
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return self.conv_cat(h_feature+low_feature)




class EDCls_UNet2(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.2):
        super(EDCls_UNet2, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.training_mode = training_mode
        if self.use_CatOut:
            logger.info("using cat_out for DS...")
            if cat_mode=='cat_feat':
                #self.conv_catf=nn.Conv2d(out_c*4,out_c,3,1,1)
                self.conv_catf = nn.Sequential(nn.Conv2d(filters[2]+filters[1]+filters[0]+filters[0], 32, 3, 1, 1),
                                               conv_act,
                                               nn.Conv2d(32, out_c, 3, 1,
                                                         1))
            else:
                self.conv_catf = nn.Conv2d(out_c * 4, out_c, 3, 1, 1)


        if use_att:
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                # self.AttFunc1 = BAM(filters[0], ds=8)
                # self.AttFunc2 = BAM(filters[1], ds=4)
                # self.AttFunc3 = BAM(filters[2], ds=2)

                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
        else:
            logger.info("no att for AttFunc4 is used...")


        if use_se:
            logger.info("use se for decoder...")
        self.decoder4 = unetUp3(filters[2]*2, filters[2],use_att=True,act_mode=act_mode,use_se=use_se)
        self.decoder3 = unetUp3(filters[1]*2, filters[1],use_att=True,act_mode=act_mode,use_se=use_se)
        self.decoder2 = unetUp3(filters[0]*2, filters[0],use_att=True,act_mode=act_mode,use_se=use_se)
        self.decoder1 =unetUp3(filters[0], filters[0],use_att=True,act_mode=act_mode,use_se=use_se)
        self.use_drop=False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)
            # self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
            #                                 conv_act,
            #                                 nn.Conv2d(32, 32, 3, padding=1),
            #                                 conv_act,
            #                                 nn.Dropout2d(drop_rate),
            #                                 nn.Conv2d(32, out_c, 1, padding=0))
            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        if self.use_drop:
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            self.dropblock,
                                            nn.Conv2d(32, out_c, 3, padding=1))
        else:
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            nn.Conv2d(32, out_c, 3, padding=1))

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        # feat12_0=F.relu(feat1_0-feat2_0)##[4,64,128,128]
        # feat12_1 = F.relu(feat1_1 - feat2_1)#[4,64,64,64]
        # feat12_2 = F.relu(feat1_2 - feat2_2)#[4,128,32,32]
        # feat12_3 = F.relu(feat1_3 - feat2_3)#[4,256,16,16]
        # feat12_4 = F.relu(feat1_4 - feat2_4)#[4,512,8,8]


        self.use_DS=use_DS
        if self.use_DS:
            # self.classifier0=nn.Sequential(nn.Conv2d(filter[0],32,3,1,1),  #64
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(32, 32, 3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(32, out_c, 3, padding=1))
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        if self.use_att:
            # =======methods1 using PAM for feat1, feat2========
            # feat1_4=self.AttFunc(feat1_4)
            # feat2_4=self.AttFunc(feat2_4)
            # ======methods2 using PAM for feat12======
            # height = feat1_1.shape[3]
            # feat12_1 = torch.cat((feat1_1, feat2_1), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_1 = self.AttFunc1(feat12_1)
            # feat1_1, feat2_1 = feat12_1[:, :, :, 0:height], feat12_1[:, :, :, height:]
            #
            # height = feat1_2.shape[3]
            # feat12_2 = torch.cat((feat1_2, feat2_2), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_2 = self.AttFunc2(feat12_2)
            # feat1_2, feat2_2 = feat12_2[:, :, :, 0:height], feat12_2[:, :, :, height:]
            #
            # height = feat1_3.shape[3]
            # feat12_3 = torch.cat((feat1_3, feat2_3), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_3 = self.AttFunc3(feat12_3)
            # feat1_3, feat2_3 = feat12_3[:, :, :, 0:height], feat12_3[:, :, :, height:]
            #===========================================
            height = feat1_4.shape[3]
            feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            feat12_4 = self.AttFunc4(feat12_4)
            feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]

        # feat12_0=F.relu(feat1_0-feat2_0)##[4,64,128,128]
        # feat12_1 = F.relu(feat1_1 - feat2_1)#[4,64,64,64]
        # feat12_2 = F.relu(feat1_2 - feat2_2)#[4,128,32,32]
        # feat12_3 = F.relu(feat1_3 - feat2_3)#[4,256,16,16]
        # feat12_4 = F.relu(feat1_4 - feat2_4)#[4,512,8,8]
        #=====F.relu may lose information=======
        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,128,128]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,64,64]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,32,32]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,16,16]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,8,8]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        # if self.use_drop:
        #     d4_1=self.dropblock(d4_1)
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        # if self.use_drop:
        #     d3_1=self.dropblock(d3_1)
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        # if self.use_drop:
        #     d2_1=self.dropblock(d2_1)
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128
        # if self.use_drop:
        #     d1_1=self.dropblock(d1_1)

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        # if self.use_drop:
        #     d4_2=self.dropblock(d4_2)
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        # if self.use_drop:
        #     d3_2=self.dropblock(d3_2)
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        # if self.use_drop:
        #     d2_2=self.dropblock(d2_2)
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128
        # if self.use_drop:
        #     d1_2=self.dropblock(d1_2)

        pred1 = self.classifier(d1_1)
        pred2 = self.classifier(d1_2)

        if self.training_mode:
            if self.use_DS:

                pred1_0, pred2_0 = self.class_conv1(d4_1), self.class_conv1(d4_2)
                pred1_1, pred2_1 = self.class_conv2(d3_1), self.class_conv2(d3_2)
                pred1_2, pred2_2 = self.class_conv3(d2_1), self.class_conv3(d2_2)

                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2)

        return  pred1,pred2








#==========================for efficient unet2===============================================================
class EDCls_UNet2_New(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.2):
        super(EDCls_UNet2_New, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        if self.use_CatOut:#====seem not to work
            logger.info("using cat_out for DS...")
            if cat_mode=='cat_feat':
                self.conv_catf = nn.Sequential(nn.Conv2d(filters[2]+filters[1]+filters[0]+filters[0], 32, 3, 1, 1),
                                               conv_act,
                                               nn.Conv2d(32, out_c, 3, 1,
                                                         1))
            else:
                self.conv_catf = nn.Conv2d(out_c * 4, out_c, 3, 1, 1)


        if use_att:#=====can be removed
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                # self.AttFunc1 = BAM(filters[0], ds=8)
                # self.AttFunc2 = BAM(filters[1], ds=4)
                # self.AttFunc3 = BAM(filters[2], ds=2)

                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
        else:
            logger.info("no att for AttFunc4 is used...")

        deconv_att=True
        if use_se:
            deconv_att=False
            logger.info("use se for decoder...")
        else:
            deconv_att = True
            logger.info("use att for decoder...")
        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        use_Py=False
        if use_Py:
            logger.info("use PyConv for decoder...")
            self.decoder4 = unetUp3_Py(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
            self.decoder3 = unetUp3_Py(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
            self.decoder2 = unetUp3_Py(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
            self.decoder1 = unetUp3_Py(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode, use_se=use_se)
        else:
            self.decoder4 = unetUp3(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se,drop_block=self.dropblock)
            self.decoder3 = unetUp3(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se,drop_block=self.dropblock)
            self.decoder2 = unetUp3(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se,drop_block=self.dropblock)
            self.decoder1 = unetUp3(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode, use_se=use_se,drop_block=self.dropblock)



        self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        # PyConv3(32, 32),
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act
                                        )
        if self.use_drop:

            self.logit_conv=nn.Sequential(
                                            # nn.Conv2d(32, 32, 3, padding=1),
                                            # conv_act,
                                            self.dropblock,
                                            nn.Conv2d(32, out_c, 3, padding=1))
        else:
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            nn.Conv2d(32, out_c, 3, padding=1))

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 #PyConv3(32, 32),
                                                 conv_act

                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 #PyConv3(32, 32),
                                                 conv_act
                                                 )
            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        if self.use_att:

            height = feat1_4.shape[3]
            feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            feat12_4 = self.AttFunc4(feat12_4)
            feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128


        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128

        pred1 = self.logit_conv(self.class_conv(d1_1))
        pred2 = self.logit_conv(self.class_conv(d1_2))

        if self.use_DS:
            pred1_0,pred2_0=self.logit_conv(self.class_conv1(d4_1)),self.logit_conv(self.class_conv1(d4_2))
            pred1_1, pred2_1 = self.logit_conv(self.class_conv2(d3_1)),self.logit_conv(self.class_conv2(d3_2))
            pred1_2, pred2_2 = self.logit_conv(self.class_conv3(d2_1)),self.logit_conv(self.class_conv3(d2_2))
            if self.use_CatOut:
                if self.cat_mode=='cat_feat':
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_1, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(d3_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True)], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_2, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(d3_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True)], dim=1))
                else:
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(pred1_0, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred1_2, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred1], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(pred2_0, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred2_1, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred2], dim=1))


                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2),(pred1_f,pred2_f)
            else:
                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2)


        return  pred1,pred2

class EDCls_UNet2_New2(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.1,se_block='BAM'):
        super(EDCls_UNet2_New2, self).__init__()
        self.feat_Extactor=net_feat
        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.training_mode=training_mode


        if use_att:#=====can be removed
            # message="using att_mode is {}".format(att_mode)
            # logger.info(message)
            # if att_mode == 'BAM':
            #     # self.AttFunc1 = BAM(filters[0], ds=8)
            #     # self.AttFunc2 = BAM(filters[1], ds=4)
            #     # self.AttFunc3 = BAM(filters[2], ds=2)
            #
            #     self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            # elif att_mode=='PCAM':
            #     self.AttFunc4=PCAM(filters[-1])
            # else:
            #     self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
            deconv_att = True
            logger.info("use att for decoder...")
        else:
            deconv_att=False
            logger.info("no att for decoder...")

        #se_block=se_block




        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        logger.info("use residual for decoder...")
        self.decoder4 = unetUp3_Res(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder3 = unetUp3_Res(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder2 = unetUp3_Res(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder1 = unetUp3_Res(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)

        # self.decoder4 = unetUp3(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder3 = unetUp3(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder2 = unetUp3(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder1 = unetUp3(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)



        self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1)
                                        )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS


        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))

        else:
            logger.info("using no deep supervsion...")

   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        # if self.use_att:
        #
        #     height = feat1_4.shape[3]
        #     feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
        #     feat12_4 = self.AttFunc4(feat12_4)
        #     feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128


        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128

        pred1 = self.class_conv(d1_1)
        pred2 = self.class_conv(d1_2)
        if self.training_mode:
            if self.use_DS:

                pred1_0, pred2_0 = self.class_conv1(d4_1), self.class_conv1(d4_2)
                pred1_1, pred2_1 = self.class_conv2(d3_1), self.class_conv2(d3_2)
                pred1_2, pred2_2 = self.class_conv3(d2_1), self.class_conv3(d2_2)

                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2)

        return  pred1,pred2
#==================for more efficient unet2 ,using res for unetup3======================
class EDCls_UNet2_Res(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.1,se_block='BAM'):
        super(EDCls_UNet2_Res, self).__init__()
        self.feat_Extactor=net_feat
        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.training_mode=training_mode


        if use_att:#=====can be removed
            # message="using att_mode is {}".format(att_mode)
            # logger.info(message)
            # if att_mode == 'BAM':
            #     # self.AttFunc1 = BAM(filters[0], ds=8)
            #     # self.AttFunc2 = BAM(filters[1], ds=4)
            #     # self.AttFunc3 = BAM(filters[2], ds=2)
            #
            #     self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            # elif att_mode=='PCAM':
            #     self.AttFunc4=PCAM(filters[-1])
            # else:
            #     self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
            pass
        else:
            logger.info("no att for AttFunc4 is used...")

        deconv_att=True
        se_block=se_block
        if use_se:
            deconv_att=False
            logger.info("use se {} for decoder...".format(se_block))
        else:
            deconv_att = True

            logger.info("use att for decoder...")
        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        logger.info("use residual for decoder...")
        self.decoder4 = unetUp3_Res(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder3 = unetUp3_Res(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder2 = unetUp3_Res(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder1 = unetUp3_Res(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)

        # self.decoder4 = unetUp3(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder3 = unetUp3(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder2 = unetUp3(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder1 = unetUp3(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)



        self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1)
                                        )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        if self.use_att:

            height = feat1_4.shape[3]
            feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            feat12_4 = self.AttFunc4(feat12_4)
            feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128


        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128

        pred1 = self.class_conv(d1_1)
        pred2 = self.class_conv(d1_2)
        if self.training_mode:
            if self.use_DS:

                pred1_0, pred2_0 = self.class_conv1(d4_1), self.class_conv1(d4_2)
                pred1_1, pred2_1 = self.class_conv2(d3_1), self.class_conv2(d3_2)
                pred1_2, pred2_2 = self.class_conv3(d2_1), self.class_conv3(d2_2)

                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2)

        return  pred1,pred2

#==========================using unet3+ like full-scale connection, seem not to work======================================
class EDCls_UNet3_Plus(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.2):
        super(EDCls_UNet3_Plus, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode


        if use_att:#=====can be removed
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                # self.AttFunc1 = BAM(filters[0], ds=8)
                # self.AttFunc2 = BAM(filters[1], ds=4)
                # self.AttFunc3 = BAM(filters[2], ds=2)

                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
        else:
            logger.info("no att for AttFunc4 is used...")

        deconv_att=True
        if use_se:
            deconv_att=False
            logger.info("use se for decoder...")
        else:
            deconv_att = True
            logger.info("use att for decoder...")
        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        self.CatChannels = filters[0]//2
        #e0
        self.e0d4_conv=nn.Sequential(
            nn.MaxPool2d(8, 8, ceil_mode=True),
            nn.Conv2d(filters[0],self.CatChannels,3,padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e0d3_conv = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e0d2_conv = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e0d1_conv = nn.Sequential(
            #nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        #e1
        self.e1d4_conv = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e1d3_conv = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e1d2_conv = nn.Sequential(
            #nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[0], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        #e2
        self.e2d4_conv = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[1], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e2d3_conv = nn.Sequential(
            #nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[1], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        #e3
        self.e3d4_conv = nn.Sequential(
            # nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(filters[2], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        #e4
        self.e4d4_conv=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(filters[3], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e4d3_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(filters[3], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e4d2_conv = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(filters[3], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.e4d1_conv = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear'),
            nn.Conv2d(filters[3], self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        #d4
        self.d4d3_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(filters[0]//2, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.d4d2_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(filters[0]//2, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.d4d1_conv = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(filters[0]//2, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        # d3
        self.d3d2_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(filters[0]//2, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        self.d3d1_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(filters[0]//2, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        # d2
        self.d2d1_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(filters[0]//2, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
        )
        #fuse
        # self.fea_fuse4=nn.Sequential(
        #     nn.Conv2d(self.CatChannels * 4, self.CatChannels, 3, padding=1),
        #     nn.BatchNorm2d(self.CatChannels),
        #     nn.ReLU(inplace=True)
        # )
        # self.fea_fuse3 = nn.Sequential(
        #     nn.Conv2d(self.CatChannels * 3, self.CatChannels, 3, padding=1),
        #     nn.BatchNorm2d(self.CatChannels),
        #     nn.ReLU(inplace=True)
        # )
        # self.fea_fuse2 = nn.Sequential(
        #     nn.Conv2d(self.CatChannels * 2, self.CatChannels, 3, padding=1),
        #     nn.BatchNorm2d(self.CatChannels),
        #     nn.ReLU(inplace=True)
        # )
        #cat
        self.cat_conv4=nn.Sequential(
            nn.Conv2d(self.CatChannels*5, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.CatChannels * 2, self.CatChannels * 2, 3, padding=1),
            # nn.BatchNorm2d(self.CatChannels * 2),
            # nn.ReLU(inplace=True)

        )
        self.cat_conv3 = nn.Sequential(
            nn.Conv2d(self.CatChannels * 5, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.CatChannels * 2, self.CatChannels * 2, 3, padding=1),
            # nn.BatchNorm2d(self.CatChannels * 2),
            # nn.ReLU(inplace=True)
        )
        self.cat_conv2 = nn.Sequential(
            nn.Conv2d(self.CatChannels * 5, self.CatChannels , 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.CatChannels * 2, self.CatChannels * 2, 3, padding=1),
            # nn.BatchNorm2d(self.CatChannels * 2),
            # nn.ReLU(inplace=True)
        )
        self.cat_conv1 = nn.Sequential(
            nn.Conv2d(self.CatChannels * 5, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels ),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.CatChannels * 2, self.CatChannels * 2, 3, padding=1),
            # nn.BatchNorm2d(self.CatChannels * 2),
            # nn.ReLU(inplace=True)
        )

        self.class_conv = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1)
                                        )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 # nn.Conv2d(64, 32, 3, 1, 1),  # 256
                                                 # conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(
                                                 # nn.Conv2d(64, 32, 3, 1, 1),  # 128
                                                 # conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(
                                                 # nn.Conv2d(64, 32, 3, 1, 1),  # 64
                                                 # conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )




   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))



        d4_1=self.cat_conv4(torch.cat([self.e4d4_conv(feat12_4),
                            self.e0d4_conv(feat1_0+feat12_0),self.e1d4_conv(feat1_1+feat12_1),
                            self.e2d4_conv(feat1_2+feat12_2),self.e3d4_conv(feat1_3+feat12_3) ],dim=1))


        d3_1 = self.cat_conv3(torch.cat([self.e4d3_conv(feat12_4),self.d4d3_conv(d4_1),
                                         self.e0d3_conv(feat1_0+feat12_0),self.e1d3_conv(feat1_1+feat12_1),
                                         self.e2d3_conv(feat1_2+feat12_2)

                                         ], dim=1))
        d2_1 = self.cat_conv2(torch.cat([self.e4d2_conv(feat12_4), self.d4d2_conv(d4_1),self.d3d2_conv(d3_1),
                                         self.e0d2_conv(feat1_0+feat12_0),self.e1d2_conv(feat1_1+feat12_1)

                                         ], dim=1))
        d1_1 = self.cat_conv2(torch.cat([self.e4d1_conv(feat12_4), self.d4d1_conv(d4_1), self.d3d1_conv(d3_1),self.d2d1_conv(d2_1),
                                         self.e0d1_conv(feat1_0+feat12_0)

                                         ], dim=1))

        #T2
        d4_2 = self.cat_conv4(torch.cat([self.e4d4_conv(feat12_4),
                                         self.e0d4_conv(feat2_0 + feat12_0), self.e1d4_conv(feat2_1 + feat12_1),
                                         self.e2d4_conv(feat2_2+feat12_2), self.e3d4_conv(feat2_3 + feat12_3)], dim=1))

        d3_2 = self.cat_conv3(torch.cat([self.e4d3_conv(feat12_4), self.d4d3_conv(d4_2),
                                         self.e0d3_conv(feat2_0 + feat12_0), self.e1d3_conv(feat2_1+feat12_1),
                                         self.e2d3_conv(feat2_2+feat12_2)
                                         ], dim=1))
        d2_2 = self.cat_conv2(torch.cat([self.e4d2_conv(feat12_4), self.d4d2_conv(d4_2), self.d3d2_conv(d3_2),
                                         self.e0d2_conv(feat2_0 + feat12_0), self.e1d2_conv(feat2_1 + feat12_1)
                                         ], dim=1))
        d1_2 = self.cat_conv2(
            torch.cat([self.e4d1_conv(feat12_4), self.d4d1_conv(d4_2), self.d3d1_conv(d3_2), self.d2d1_conv(d2_2),
                       self.e0d1_conv(feat2_0 + feat12_0)
                       ], dim=1))




        # d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        # d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        # d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        # d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128
        #
        # d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        # d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        # d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        # d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128



        pred1=self.class_conv(d1_1)
        pred2=self.class_conv(d1_2)

        # pred1 = self.logit_conv(self.class_conv(d1_1))
        # pred2 = self.logit_conv(self.class_conv(d1_2))

        if self.use_DS:
            # pred1_0,pred2_0=self.logit_conv(self.class_conv1(d4_1)),self.logit_conv(self.class_conv1(d4_2))
            # pred1_1, pred2_1 = self.logit_conv(self.class_conv2(d3_1)),self.logit_conv(self.class_conv2(d3_2))
            # pred1_2, pred2_2 = self.logit_conv(self.class_conv3(d2_1)),self.logit_conv(self.class_conv3(d2_2))

            pred1_0, pred2_0 = self.class_conv1(d4_1), self.class_conv1(d4_2)
            pred1_1, pred2_1 = self.class_conv2(d3_1), self.class_conv2(d3_2)
            pred1_2, pred2_2 = self.class_conv3(d2_1),self.class_conv3(d2_2)


            return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2)


        return  pred1,pred2
#==============================using five dedcoders================================
class EDCls_UNet2_New5(nn.Module):#seem not to be better than EDCls_UNet2_New, it needs more params for decoders

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.2):
        super(EDCls_UNet2_New5, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        if self.use_CatOut:#====seem not to work
            logger.info("using cat_out for DS...")
            if cat_mode=='cat_feat':
                self.conv_catf = nn.Sequential(nn.Conv2d(filters[2]+filters[1]+filters[0]+filters[0], 32, 3, 1, 1),
                                               conv_act,
                                               nn.Conv2d(32, out_c, 3, 1,
                                                         1))
            else:
                self.conv_catf = nn.Conv2d(out_c * 4, out_c, 3, 1, 1)

        if use_att:#=====can be removed
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                # self.AttFunc1 = BAM(filters[0], ds=8)
                # self.AttFunc2 = BAM(filters[1], ds=4)
                # self.AttFunc3 = BAM(filters[2], ds=2)

                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
        else:
            logger.info("no att for AttFunc4 is used...")

        deconv_att=True
        if use_se:
            deconv_att=False
            logger.info("use se for decoder...")
        self.decoder4 = unetUp3(filters[2], filters[2],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder3 = unetUp3(filters[1]*2, filters[1],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder2 = unetUp3(filters[0]*2, filters[0],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder1 =unetUp3(filters[0], filters[0],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder0=unetUp1_64(filters[0],32,act_mode=act_mode,use_se=use_se)
        self.use_drop=False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        if self.use_drop:
            self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            # nn.Conv2d(32, 32, 3, padding=1),
                                            # conv_act,
                                            # self.dropblock,
                                            # nn.Conv2d(32, out_c, 3, padding=1)
                                            )
            self.logit_conv=nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            self.dropblock,
                                            nn.Conv2d(32, out_c, 3, padding=1))
        else:
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            nn.Conv2d(32, out_c, 3, padding=1))

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:

            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 # nn.Conv2d(32, 32, 3, padding=1),
                                                 # conv_act,
                                                 # self.dropblock,
                                                 # nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 # nn.Conv2d(32, 32, 3, padding=1),
                                                 # conv_act,
                                                 # self.dropblock,
                                                 # nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 # nn.Conv2d(32, 32, 3, padding=1),
                                                 # conv_act,
                                                 # self.dropblock,
                                                 # nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        if self.use_att:

            height = feat1_4.shape[3]
            feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            feat12_4 = self.AttFunc4(feat12_4)
            feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128
        d0_1=self.decoder0(d1_1)

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128
        d0_2=self.decoder0(d1_2)


        pred1 = self.logit_conv(d0_1)
        pred2 = self.logit_conv(d0_2)


        if self.use_DS:
            pred1_0,pred2_0=self.logit_conv(self.class_conv1(d4_1)),self.logit_conv(self.class_conv1(d4_2))
            pred1_1, pred2_1 = self.logit_conv(self.class_conv2(d3_1)),self.logit_conv(self.class_conv2(d3_2))
            pred1_2, pred2_2 = self.logit_conv(self.class_conv3(d2_1)),self.logit_conv(self.class_conv3(d2_2))
            pred1_3, pred2_3 = self.logit_conv(self.class_conv3(d1_1)), self.logit_conv(self.class_conv3(d1_2))
            if self.use_CatOut:
                if self.cat_mode=='cat_feat':
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_1, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(d3_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True)], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_2, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(d3_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True)], dim=1))
                else:
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(pred1_0, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred1_2, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred1], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(pred2_0, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred2_1, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred2], dim=1))


                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2),(pred1_f,pred2_f)
            else:
                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1_3, pred2_3),(pred1, pred2)


        return  pred1,pred2

#=============================for serestnet50-backbone========================
class EDCls_UNet2_Res50(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.2):
        super(EDCls_UNet2_Res50, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        if self.use_CatOut:#====seem not to work
            logger.info("using cat_out for DS...")
            if cat_mode=='cat_feat':
                self.conv_catf = nn.Sequential(nn.Conv2d(filters[2]+filters[1]+filters[0]+filters[0], 32, 3, 1, 1),
                                               conv_act,
                                               nn.Conv2d(32, out_c, 3, 1,
                                                         1))
            else:
                self.conv_catf = nn.Conv2d(out_c * 4, out_c, 3, 1, 1)


        if use_att:#=====can be removed
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                # self.AttFunc1 = BAM(filters[0], ds=8)
                # self.AttFunc2 = BAM(filters[1], ds=4)
                # self.AttFunc3 = BAM(filters[2], ds=2)

                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
        else:
            logger.info("no att for AttFunc4 is used...")

        deconv_att=True
        if use_se:
            deconv_att=False
            logger.info("use se for decoder...")

        # conv1 = self.conv1(inputs)  # 1/2 [1,3,512,512]==>[1,64,256,256]
        # conv2 = self.conv2(conv1)  # 1/2  [1,256,256,256]
        # conv3 = self.conv3(conv2)  # 1/4  [1,512,128,128]
        # conv4 = self.conv4(conv3)  # 1/8  [1,1024,64,64]
        # conv5 = self.conv5(conv4)  # 1/16  [1,2048,32,32]
        # center = self.center(conv5)  # [1,256,16,16]  up_in, x_in, n_out
        self.decoder4 = unetUp3_50(512,2048,256,use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder3 = unetUp3_50(256,1024,128, filters[1],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder2 = unetUp3_50(128,512,64, filters[0],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder1 =unetUp3_50(64,256,64, filters[0],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder0=unetUp3_50(64,0,32,act_mode=act_mode,use_se=use_se)
        self.use_drop=False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        if self.use_drop:
            self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            # nn.Conv2d(32, 32, 3, padding=1),
                                            # conv_act,
                                            # self.dropblock,
                                            # nn.Conv2d(32, out_c, 3, padding=1)
                                            )
            self.logit_conv=nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            self.dropblock,
                                            nn.Conv2d(32, out_c, 3, padding=1))
        else:
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            nn.Conv2d(32, out_c, 3, padding=1))

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:

            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 # nn.Conv2d(32, 32, 3, padding=1),
                                                 # conv_act,
                                                 # self.dropblock,
                                                 # nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 # nn.Conv2d(32, 32, 3, padding=1),
                                                 # conv_act,
                                                 # self.dropblock,
                                                 # nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 # nn.Conv2d(32, 32, 3, padding=1),
                                                 # conv_act,
                                                 # self.dropblock,
                                                 # nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        if self.use_att:

            height = feat1_4.shape[3]
            feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            feat12_4 = self.AttFunc4(feat12_4)
            feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,128,128]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,64,64]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,32,32]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,16,16]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,8,8]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128
        d0_1=self.decoder0(d1_1)

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128
        d0_2=self.decoder0(d1_2)

        # pred1 = self.logit_conv(self.class_conv(d1_1))
        # pred2 = self.logit_conv(self.class_conv(d1_2))
        pred1 = self.logit_conv(d0_1)
        pred2 = self.logit_conv(d0_2)


        if self.use_DS:
            pred1_0,pred2_0=self.logit_conv(self.class_conv1(d4_1)),self.logit_conv(self.class_conv1(d4_2))
            pred1_1, pred2_1 = self.logit_conv(self.class_conv2(d3_1)),self.logit_conv(self.class_conv2(d3_2))
            pred1_2, pred2_2 = self.logit_conv(self.class_conv3(d2_1)),self.logit_conv(self.class_conv3(d2_2))
            pred1_3, pred2_3 = self.logit_conv(self.class_conv3(d1_1)), self.logit_conv(self.class_conv3(d1_2))
            if self.use_CatOut:
                if self.cat_mode=='cat_feat':
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_1, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(d3_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True)], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_2, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(d3_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True)], dim=1))
                else:
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(pred1_0, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred1_2, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred1], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(pred2_0, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred2_1, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred2], dim=1))


                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2),(pred1_f,pred2_f)
            else:
                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1_3, pred2_3),(pred1, pred2)


        return  pred1,pred2


#=========================================================================================
class EDCls_UNet2_DiffAdd(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.2):
        super(EDCls_UNet2_DiffAdd, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        if self.use_CatOut:
            logger.info("using cat_out for DS...")
            if cat_mode=='cat_feat':
                #self.conv_catf=nn.Conv2d(out_c*4,out_c,3,1,1)
                self.conv_catf = nn.Sequential(nn.Conv2d(filters[2]+filters[1]+filters[0]+filters[0], 32, 3, 1, 1),
                                               conv_act,
                                               nn.Conv2d(32, out_c, 3, 1,
                                                         1))
            else:
                self.conv_catf = nn.Conv2d(out_c * 4, out_c, 3, 1, 1)


        if use_att:
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                # self.AttFunc1 = BAM(filters[0], ds=8)
                # self.AttFunc2 = BAM(filters[1], ds=4)
                # self.AttFunc3 = BAM(filters[2], ds=2)

                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)


        if use_se:
            logger.info("use se for decoder...")
        self.decoder4 = unetUp3(filters[2]*2, filters[2],use_att=use_att,act_mode=act_mode,use_se=use_se)
        self.decoder3 = unetUp3(filters[1]*2, filters[1],use_att=use_att,act_mode=act_mode,use_se=use_se)
        self.decoder2 = unetUp3(filters[0]*2, filters[0],use_att=use_att,act_mode=act_mode,use_se=use_se)
        self.decoder1 =unetUp3(filters[0], filters[0],use_att=use_att,act_mode=act_mode,use_se=use_se)

        if drop_rate>0:
            #print("using drop %.3f befoe classifier" % drop_rate)
            message="using drop {:.3f} before classifier".format(drop_rate)
            logger.info(message)
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            nn.Dropout2d(drop_rate),
                                            nn.Conv2d(32, out_c, 1, padding=0))
        else:
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            nn.Conv2d(32, out_c, 3, padding=1))
        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        if diff_mode=='diffadd':
            self.conv_cat01 = nn.Conv2d(filters[0], filters[0], 1)
            self.conv_cat2 = nn.Conv2d(filters[1], filters[1], 1)
            self.conv_cat3 = nn.Conv2d(filters[2], filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3], filters[3], 1)
        # feat12_0=F.relu(feat1_0-feat2_0)##[4,64,128,128]
        # feat12_1 = F.relu(feat1_1 - feat2_1)#[4,64,64,64]
        # feat12_2 = F.relu(feat1_2 - feat2_2)#[4,128,32,32]
        # feat12_3 = F.relu(feat1_3 - feat2_3)#[4,256,16,16]
        # feat12_4 = F.relu(feat1_4 - feat2_4)#[4,512,8,8]


        self.use_DS=use_DS
        if self.use_DS:
            # self.classifier0=nn.Sequential(nn.Conv2d(filter[0],32,3,1,1),  #64
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(32, 32, 3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(32, out_c, 3, padding=1))
            logger.info("using deep supervsion...")
            if drop_rate>0:

                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Dropout2d(drop_rate),
                                                 nn.Conv2d(32, out_c, 1, padding=0))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Dropout2d(drop_rate),
                                                 nn.Conv2d(32, out_c, 1, padding=0))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Dropout2d(drop_rate),
                                                 nn.Conv2d(32, out_c, 1, padding=0))
            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))


   def forward(self, x1, x2):
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        if self.use_att:
            # =======methods1 using PAM for feat1, feat2========
            # feat1_4=self.AttFunc(feat1_4)
            # feat2_4=self.AttFunc(feat2_4)
            # ======methods2 using PAM for feat12======
            # height = feat1_1.shape[3]
            # feat12_1 = torch.cat((feat1_1, feat2_1), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_1 = self.AttFunc1(feat12_1)
            # feat1_1, feat2_1 = feat12_1[:, :, :, 0:height], feat12_1[:, :, :, height:]
            #
            # height = feat1_2.shape[3]
            # feat12_2 = torch.cat((feat1_2, feat2_2), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_2 = self.AttFunc2(feat12_2)
            # feat1_2, feat2_2 = feat12_2[:, :, :, 0:height], feat12_2[:, :, :, height:]
            #
            # height = feat1_3.shape[3]
            # feat12_3 = torch.cat((feat1_3, feat2_3), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_3 = self.AttFunc3(feat12_3)
            # feat1_3, feat2_3 = feat12_3[:, :, :, 0:height], feat12_3[:, :, :, height:]
            #===========================================
            height = feat1_4.shape[3]
            feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            feat12_4 = self.AttFunc4(feat12_4)
            feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]

        # feat12_0=F.relu(feat1_0-feat2_0)##[4,64,128,128]
        # feat12_1 = F.relu(feat1_1 - feat2_1)#[4,64,64,64]
        # feat12_2 = F.relu(feat1_2 - feat2_2)#[4,128,32,32]
        # feat12_3 = F.relu(feat1_3 - feat2_3)#[4,256,16,16]
        # feat12_4 = F.relu(feat1_4 - feat2_4)#[4,512,8,8]
        #=====F.relu may lose information=======
        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,128,128]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,64,64]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,32,32]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,16,16]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,8,8]
        elif self.diff_mode=='cat':
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))
        else:
            #using both diff and add feature
            feat12_0=torch.abs(feat1_0-feat2_0)+self.conv_cat01(torch.abs(feat1_0+feat2_0))
            feat12_1 = torch.abs(feat1_1 - feat2_1) + self.conv_cat01(torch.abs(feat1_1 + feat2_1))
            feat12_2 = torch.abs(feat1_2 - feat2_2) + self.conv_cat2(torch.abs(feat1_2 + feat2_2))
            feat12_3 = torch.abs(feat1_3 - feat2_3) + self.conv_cat3(torch.abs(feat1_3 + feat2_3))
            feat12_4 = torch.abs(feat1_4 - feat2_4) + self.conv_cat4(torch.abs(feat1_4 + feat2_4))




        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128

        pred1 = self.classifier(d1_1)
        pred2 = self.classifier(d1_2)

        if self.use_DS:
            pred1_0,pred2_0=self.classifier1(d4_1), self.classifier1(d4_2)
            pred1_1, pred2_1 = self.classifier2(d3_1), self.classifier2(d3_2)
            pred1_2, pred2_2 = self.classifier3(d2_1), self.classifier3(d2_2)
            if self.use_CatOut:
                if self.cat_mode=='cat_feat':
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_1, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(d3_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True)], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_2, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(d3_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True)], dim=1))
                else:
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(pred1_0, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred1_2, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred1], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(pred2_0, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred2_1, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred2], dim=1))


                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2),(pred1_f,pred2_f)
            else:
                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2)


        return  pred1,pred2

#=======================for DSCD===========================================
class EDCls_UNet_DSCD(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.1,se_block='BAM'):
        super(EDCls_UNet_DSCD, self).__init__()

        self.in_c=in_c
        filters=[32,64,128,256,512]
        if act_mode=='relu':
            logger.info("using act mode {}".format(act_mode))
            conv_act=nn.ReLU(inplace=True)
        else:
            logger.info("using act mode {}".format(act_mode))
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        #====================for encoder=========================
        self.is_batchnorm=True
        self.conv1 = unetConv2_res(self.in_c*2, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2_res(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2_res(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2_res(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2_res(filters[3], filters[4], self.is_batchnorm)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.center=Dblock(512)
        #=========================for decoder=====================================

        deconv_att=True

        self.decoder5= unetUp2(filters[4], filters[4], use_att=deconv_att, act_mode=act_mode)
        self.decoder4= unetUp2(filters[4], filters[3], use_att=deconv_att, act_mode=act_mode)
        self.decoder3= unetUp2(filters[3], filters[2], use_att=deconv_att, act_mode=act_mode)
        self.decoder2= unetUp2(filters[2], filters[1], use_att=deconv_att, act_mode=act_mode)
        self.decoder1 = unetUp2(filters[1], filters[0], use_att=deconv_att, act_mode=act_mode)
        if drop_rate > 0:
            self.use_drop = True
            message = "using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)
            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1.5e4
            )
        self.class_conv = nn.Sequential(
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1),

                                        )



   def forward(self, x1):#x1=img1+cd x2=img2+cd
        if self.use_drop:
            self.dropblock.step()

        conv1_1 = self.conv1(x1)  # [1,4,512,512]==>[1,32,512,512]
        maxpool1_1 = self.maxpool1(conv1_1)  # [1,32,512,512]==>[1,32,256,256]
        conv2_1 = self.conv2(maxpool1_1)  # [1,32,256,256]==>[1,64,256,256]
        maxpool2_1 = self.maxpool2(conv2_1)  # [1,64,256,256]==>[1,64,128,128]
        conv3_1 = self.conv3(maxpool2_1)  # [1,64,128,128]==>[1,128,128,128]
        maxpool3_1 = self.maxpool3(conv3_1)  # [1,128,128,128]==>[1,128,64,64]
        conv4_1 = self.conv4(maxpool3_1)  # [1,128,64,64]==>[1,256,64,64]
        maxpool4_1 = self.maxpool4(conv4_1)  # [1,256,64,64]==>[1,256,32,32]
        conv5_1 = self.conv5(maxpool4_1)  # [1,256,32,32]==>[1,512,32,32]
        maxpool5_1 = self.maxpool5(conv5_1)  # [1,512,32,32]==>[1,512,16,16]
        center1 = self.center(maxpool5_1)  # [1,512,16,16]==>[1,512,16,16]



        #===========================for decoder====================================
        d5_1=self.decoder5(center1,conv5_1)#[4,512,16,16] [4,512,32,32]
        d4_1=self.decoder4(d5_1,conv4_1)
        d3_1=self.decoder3(d4_1,conv3_1)
        d2_1=self.decoder2(d3_1,conv2_1)#[4,128,128,128] [4,64,256,256]
        d1_1=self.decoder1(d2_1,conv1_1)
        pred1=self.class_conv(d1_1)
        return  pred1





















#=====================for DS with rgb255 guidance, seems not to improve===============
class EDCls_UNet3(nn.Module):
   #=====using binary change map guidance to improve multi-class segmentation=======================
   def __init__(self, in_c=3, use_att=False,diff_mode='diff', att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,drop_rate=0.2):
        super(EDCls_UNet3, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            logger.info("using act mode {}".format(act_mode))
            conv_act=nn.ReLU(inplace=True)
        else:
            logger.info("using act mode {}".format(act_mode))
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        if use_att:
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
        else:
            logger.info("no att for AttFunc4 is used...")


        deconv_att=True
        self.decoder4 = unetUp3(filters[2]*2, filters[2],use_att=deconv_att,act_mode=act_mode)
        self.decoder3= unetUp3(filters[1]*2, filters[1],use_att=deconv_att,act_mode=act_mode)
        self.decoder2= unetUp3(filters[0]*2, filters[0],use_att=deconv_att,act_mode=act_mode)
        self.decoder1 =unetUp3(filters[0], filters[0],use_att=deconv_att,act_mode=act_mode)

        self.decoder4_diff = unetUp2(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode)
        self.decoder3_diff = unetUp2(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode)
        self.decoder2_diff = unetUp2(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode)
        self.decoder1_diff = unetUp2(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode)

        self.final_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act)
        self.classifier = nn.Conv2d(32, out_c, 3, padding=1)
        self.classifier12 = nn.Conv2d(32, 1, 3, padding=1)

        if drop_rate > 0:
            self.use_drop = True
            message = "using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1.5e4
            )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)

        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")
            self.final_conv1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                             conv_act,
                                             nn.Conv2d(32, 32, 3, padding=1),
                                             conv_act)

            self.final_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 256
                                             conv_act,
                                             nn.Conv2d(32, 32, 3, padding=1),
                                             conv_act)

            self.final_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 256
                                             conv_act,
                                             nn.Conv2d(32, 32, 3, padding=1),
                                             conv_act)

   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)

        #=====F.relu may lose information=======
        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,128,128]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,64,64]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,32,32]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,16,16]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,8,8]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))

        d4_1 = self.decoder4(feat12_4, feat12_3, feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1, feat12_2, feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1, feat12_1, feat1_1)  # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1, feat12_0, feat1_0)  # 128

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128


        d4=self.decoder4_diff(feat12_4,feat12_3)
        d3 = self.decoder3_diff(d4,feat12_2)
        d2 = self.decoder2_diff(d3,feat12_1)
        d1=self.decoder1_diff(d2,feat12_0)

        if self.use_drop:
            pred1 = self.classifier(self.dropblock(self.final_conv(d1_1)))
            pred2 = self.classifier(self.dropblock(self.final_conv(d1_2)))
            pred12 = self.classifier12(self.dropblock(self.final_conv(d1)))
        else:
            pred1 = self.classifier(self.final_conv(d1_1))
            pred2 = self.classifier(self.final_conv(d1_2))
            pred12 = self.classifier12(self.final_conv(d1))

        if self.use_DS:
            if self.use_drop:
                pred12_0 = self.classifier12(self.dropblock(self.final_conv1(d4)))
                pred12_1 = self.classifier12(self.dropblock(self.final_conv2(d3)))
                pred12_2 = self.classifier12(self.dropblock(self.final_conv3(d2)))

                pred1_0, pred2_0 = self.classifier(self.dropblock(self.final_conv1(d4_1))), self.classifier(self.dropblock(self.final_conv1(d4_2)))
                pred1_1, pred2_1 = self.classifier(self.dropblock(self.final_conv2(d3_1))), self.classifier(self.dropblock(self.final_conv2(d3_2)))
                pred1_2, pred2_2 = self.classifier(self.dropblock(self.final_conv3(d2_1))), self.classifier(self.dropblock(self.final_conv3(d2_2)))
            else:
                pred12_0 = self.classifier12(self.final_conv1(d4))
                pred12_1 = self.classifier12(self.final_conv2(d3))
                pred12_2 = self.classifier12(self.final_conv3(d2))

                pred1_0, pred2_0 = self.classifier(self.final_conv1(d4_1)), self.classifier(self.final_conv1(d4_2))
                pred1_1, pred2_1 = self.classifier(self.final_conv2(d3_1)), self.classifier(self.final_conv2(d3_2))
                pred1_2, pred2_2 = self.classifier(self.final_conv3(d2_1)), self.classifier(self.final_conv3(d2_2))


            return (pred1_0,pred2_0,F.sigmoid(pred12_0)),(pred1_1,pred2_1,F.sigmoid(pred12_1)),(pred1_2,pred2_2,F.sigmoid(pred12_2)),(pred1,pred2,F.sigmoid(pred12))

        return  pred1,pred2,F.sigmoid(pred12)

#========================for DS with hypercolumn feat=================================
class EDCls_UNet4(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.2):
        super(EDCls_UNet4, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.use_drop = False
        if drop_rate > 0:
            self.use_drop = True
            message = "using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        if self.use_CatOut:
            logger.info("using cat_out for DS...")
            if cat_mode=='cat_feat':
                #self.conv_catf=nn.Conv2d(out_c*4,out_c,3,1,1)
                self.conv_catf = nn.Sequential(nn.Conv2d(64*4, 32, 3, 1, 1),
                                               conv_act,
                                               self.dropblock,
                                               nn.Conv2d(32, out_c, 3, 1,
                                                         1))
            else:
                self.conv_catf = nn.Conv2d(out_c * 4, out_c, 3, 1, 1)


        if use_att:
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                # self.AttFunc1 = BAM(filters[0], ds=8)
                # self.AttFunc2 = BAM(filters[1], ds=4)
                # self.AttFunc3 = BAM(filters[2], ds=2)

                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
        else:
            logger.info("no att for AttFunc4 is used...")


        if use_se:
            logger.info("use se for decoder...")
        self.decoder4 = unetUp3_64(filters[3], filters[2],use_att=True,act_mode=act_mode,use_se=use_se)
        self.decoder3 = unetUp3_64(64, filters[1],use_att=True,act_mode=act_mode,use_se=use_se)
        self.decoder2 = unetUp3_64(64, filters[0],use_att=True,act_mode=act_mode,use_se=use_se)
        self.decoder1 =unetUp3_64(64, filters[0],use_att=True,act_mode=act_mode,use_se=use_se)

        if self.use_drop:
            self.classifier = nn.Sequential(
                                            # nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            # conv_act,
                                            nn.Conv2d(64, 32, 3, padding=1),
                                            conv_act,
                                            self.dropblock,
                                            nn.Conv2d(32, out_c, 3, padding=1))
        else:
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            nn.Conv2d(32, out_c, 3, padding=1))

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            # self.classifier0=nn.Sequential(nn.Conv2d(filter[0],32,3,1,1),  #64
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(32, 32, 3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(32, out_c, 3, padding=1))
            logger.info("using deep supervsion...")




   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        if self.use_att:
            # =======methods1 using PAM for feat1, feat2========
            # feat1_4=self.AttFunc(feat1_4)
            # feat2_4=self.AttFunc(feat2_4)
            # ======methods2 using PAM for feat12======
            # height = feat1_1.shape[3]
            # feat12_1 = torch.cat((feat1_1, feat2_1), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_1 = self.AttFunc1(feat12_1)
            # feat1_1, feat2_1 = feat12_1[:, :, :, 0:height], feat12_1[:, :, :, height:]
            #
            # height = feat1_2.shape[3]
            # feat12_2 = torch.cat((feat1_2, feat2_2), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_2 = self.AttFunc2(feat12_2)
            # feat1_2, feat2_2 = feat12_2[:, :, :, 0:height], feat12_2[:, :, :, height:]
            #
            # height = feat1_3.shape[3]
            # feat12_3 = torch.cat((feat1_3, feat2_3), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            # feat12_3 = self.AttFunc3(feat12_3)
            # feat1_3, feat2_3 = feat12_3[:, :, :, 0:height], feat12_3[:, :, :, height:]
            #===========================================
            height = feat1_4.shape[3]
            feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            feat12_4 = self.AttFunc4(feat12_4)
            feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]

        # feat12_0=F.relu(feat1_0-feat2_0)##[4,64,128,128]
        # feat12_1 = F.relu(feat1_1 - feat2_1)#[4,64,64,64]
        # feat12_2 = F.relu(feat1_2 - feat2_2)#[4,128,32,32]
        # feat12_3 = F.relu(feat1_3 - feat2_3)#[4,256,16,16]
        # feat12_4 = F.relu(feat1_4 - feat2_4)#[4,512,8,8]
        #=====F.relu may lose information=======
        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,128,128]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,64,64]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,32,32]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,16,16]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,8,8]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        # if self.use_drop:
        #     d4_1=self.dropblock(d4_1)
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        # if self.use_drop:
        #     d3_1=self.dropblock(d3_1)
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        # if self.use_drop:
        #     d2_1=self.dropblock(d2_1)
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128
        # if self.use_drop:
        #     d1_1=self.dropblock(d1_1)

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        # if self.use_drop:
        #     d4_2=self.dropblock(d4_2)
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        # if self.use_drop:
        #     d3_2=self.dropblock(d3_2)
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        # if self.use_drop:
        #     d2_2=self.dropblock(d2_2)
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128
        # if self.use_drop:
        #     d1_2=self.dropblock(d1_2)

        pred1 = self.classifier(d1_1)
        pred2 = self.classifier(d1_2)

        if self.use_DS:
            pred1_0,pred2_0=self.classifier(d4_1), self.classifier(d4_2)
            pred1_1, pred2_1 = self.classifier(d3_1), self.classifier(d3_2)
            pred1_2, pred2_2 = self.classifier(d2_1), self.classifier(d2_2)
            if self.use_CatOut:
                if self.cat_mode=='cat_feat':
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_1, (512,512), mode='bilinear', align_corners=True),
                                   F.interpolate(d3_1, (512,512), mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_1, (512,512), mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_1, (512,512), mode='bilinear',
                                                 align_corners=True)], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_2, (512,512), mode='bilinear', align_corners=True),
                                   F.interpolate(d3_2,(512,512), mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_2, (512,512), mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_2, (512,512), mode='bilinear',
                                                 align_corners=True)], dim=1))
                else:
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(pred1_0, pred1.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred1_1, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred1_2, pred1.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred1], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(pred2_0, pred2.size()[2:], mode='bilinear', align_corners=True),
                                   F.interpolate(pred2_1, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(pred2_2, pred2.size()[2:], mode='bilinear',
                                                 align_corners=True),
                                   pred2], dim=1))


                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2),(pred1_f,pred2_f)
            else:
                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1, pred2)


        return  pred1,pred2
#==========================for DS with hypercolumn feat, scSE elu bofore classifier================
class EDCls_UNet5(nn.Module):#consume too much GPU memory

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.2):
        super(EDCls_UNet5, self).__init__()
        self.feat_Extactor=net_feat

        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.use_drop = False
        if drop_rate > 0:
            self.use_drop = True
            message = "using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        if self.use_CatOut:
            logger.info("using cat_out for DS...")
            if cat_mode=='cat_feat':
                #self.conv_catf=nn.Conv2d(out_c*4,out_c,3,1,1)
                self.conv_catf = nn.Sequential(nn.Conv2d(64*5, 32, 3, 1, 1),
                                               #conv_act,
                                               nn.ELU(True),
                                               self.dropblock,
                                               nn.Conv2d(32, out_c, 1, padding=0))

            else:
                self.conv_catf = nn.Conv2d(out_c * 4, out_c, 3, 1, 1)


        if use_att:
            message="using att_mode is {}".format(att_mode)
            logger.info(message)
            if att_mode == 'BAM':
                # self.AttFunc1 = BAM(filters[0], ds=8)
                # self.AttFunc2 = BAM(filters[1], ds=4)
                # self.AttFunc3 = BAM(filters[2], ds=2)

                self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            elif att_mode=='PCAM':
                self.AttFunc4=PCAM(filters[-1])
            else:
                self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
        else:
            logger.info("no att for AttFunc4 is used...")

        deconv_att=True
        if use_se:
            deconv_att=False
            logger.info("use se for decoder...")
        self.decoder4 = unetUp3_64(filters[3], filters[2],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder3 = unetUp3_64(64, filters[1],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder2 = unetUp3_64(64, filters[0],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder1 =unetUp3_64(64, filters[0],use_att=deconv_att,act_mode=act_mode,use_se=use_se)
        self.decoder0=unetUp1_64(64,filters[0],act_mode=act_mode,use_se=use_se)

        if self.use_drop:
            # self.logit = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            #                            nn.ELU(True),
            #                            nn.Conv2d(64, 1, kernel_size=1, bias=False))
            # self.classifier = nn.Sequential(
            #                                 nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
            #                                 conv_act,
            #                                 nn.Conv2d(32, 32, 3, padding=1),
            #                                 nn.ELU(True),
            #                                 self.dropblock,
            #                                 nn.Conv2d(32, out_c, 1, padding=0))
            self.classifier= nn.Sequential(

                nn.Conv2d(64, 32, 3, padding=1),
                #conv_act,
                nn.ELU(True),
                self.dropblock,
                nn.Conv2d(32, out_c, 1, padding=0))
        else:
            self.classifier = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                            conv_act,
                                            nn.Conv2d(32, 32, 3, padding=1),
                                            conv_act,
                                            nn.Conv2d(32, out_c, 3, padding=1))

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")




   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        if self.use_att:

            height = feat1_4.shape[3]
            feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
            feat12_4 = self.AttFunc4(feat12_4)
            feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]


        #=====F.relu may lose information=======
        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,128,128]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,64,64]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,32,32]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,16,16]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,8,8]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128
        d0_1=self.decoder0(d1_1)

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128
        d0_2=self.decoder0(d1_2)

        # pred1 = self.classifier(d1_1)
        # pred2 = self.classifier(d1_2)

        if self.use_DS:
            pred1_0,pred2_0=self.classifier(d4_1), self.classifier(d4_2)
            pred1_1, pred2_1 = self.classifier(d3_1), self.classifier(d3_2)
            pred1_2, pred2_2 = self.classifier(d2_1), self.classifier(d2_2)
            pred1_3, pred2_3 = self.classifier(d1_1), self.classifier(d1_2)
            if self.use_CatOut:
                if self.cat_mode=='cat_feat':
                    pred1_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_1, (512,512), mode='bilinear', align_corners=True),
                                   F.interpolate(d3_1, (512,512), mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_1, (512,512), mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_1, (512,512), mode='bilinear',
                                                 align_corners=True),
                                   d0_1
                                   ], dim=1))

                    pred2_f = self.conv_catf(
                        torch.cat([F.interpolate(d4_2, (512,512), mode='bilinear', align_corners=True),
                                   F.interpolate(d3_2,(512,512), mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d2_2, (512,512), mode='bilinear',
                                                 align_corners=True),
                                   F.interpolate(d1_2, (512,512), mode='bilinear',
                                                 align_corners=True),
                                   d0_2
                                   ], dim=1))
                else:
                   pass

                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1_3, pred2_3),(pred1_f,pred2_f)
            else:
                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), (pred1_3, pred2_3)


#=====================================================================================
#===================================for BCD Model=====================================
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2,use_att=False):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))
        #self.att=SCAttention(in_size + (n_concat - 2) * out_size)
        #self.att =ChannelAttention(in_size + (n_concat - 2) * out_size)
        self.use_att=use_att
        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        if self.use_att:
            outputs0=self.att(outputs0)*outputs0
        return self.conv(outputs0)

class UNet_Nested3(nn.Module):

    def __init__(self, in_channels=6, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True,
                 use_ae=False):
        super(UNet_Nested3, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.n_classes = n_classes
        self.use_ae = use_ae
        ae_out_channels = 6

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)#use att only in the encoder part
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # self.final_1_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        # self.final_2_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        # self.final_3_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        # self.final_4_ae = nn.Conv2d(filters[0], ae_out_channels, 1)

        self.final_conv = nn.Conv2d(4 * n_classes, n_classes, 1)
        #self.final_conv_ae = nn.Conv2d(4 * ae_out_channels, ae_out_channels, 1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def set_supervised(self, value):
        self.use_ae = not value

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        X_30 = self.conv30(maxpool2)  # 128*64*64
        maxpool3 = self.maxpool(X_30)  # 128*32*32
        X_40 = self.conv40(maxpool3)  # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        if self.use_ae:
            final_1 = self.final_1_ae(X_01)
            final_2 = self.final_2_ae(X_02)
            final_3 = self.final_3_ae(X_03)
            final_4 = self.final_4_ae(X_04)

            #final = (final_1 + final_2 + final_3 + final_4) / 4
            final_cat=torch.cat((final_1,final_2,final_3,final_4),dim=1)
            final_out_ae=self.final_conv_ae(final_cat)

            return final_out_ae
        else:
            # final layer
            final_1 = self.final_1(X_01)
            final_2 = self.final_2(X_02)
            final_3 = self.final_3(X_03)
            final_4 = self.final_4(X_04)

            # final_cat = torch.cat((final_1, final_2, final_3, final_4), dim=1)
            # final_out= self.final_conv(final_cat)
            #
            # if self.n_classes==1:
            #     return F.sigmoid(final_out)
            # else:
            #     return final_out

            final = (final_1 + final_2 + final_3 + final_4) / 4

            if self.n_classes == 1:
                if self.is_ds:
                    return F.sigmoid(final)
                else:
                    return F.sigmoid(final_4)
            else:
                if self.is_ds:
                    return final
                else:
                    return final_4

#===========================for UNet++==============================
class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch,use_res=True):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.use_res=use_res

    def forward(self, x):
        if self.use_res:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)
            x0 = x
            x = self.conv2(x)
            x = self.bn2(x)
            output = self.activation(x)+x0
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.bn2(x)
            output = self.activation(x)

        return output
class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=6, out_ch=1,use_deconv=False):
        super(NestedUNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_deconv=use_deconv
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.Up0_1=nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        # self.Up1_1 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        # self.Up0_2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        # self.Up2_1 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        # self.Up1_2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        # self.Up0_3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        # self.Up3_1 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        # self.Up2_2 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        # self.Up1_3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        # self.Up0_4 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        #======for consistency with nn.Upsample============
        self.Up0_1 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        self.Up1_1 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        self.Up0_2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        self.Up2_1 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2)
        self.Up1_2 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        self.Up0_3 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        self.Up3_1 = nn.ConvTranspose2d(filters[4], filters[4], kernel_size=2, stride=2)
        self.Up2_2 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2)
        self.Up1_3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        self.Up0_4 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)


        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        if self.use_deconv:
            x0_1 = self.conv0_1(torch.cat([x0_0, self.Up0_1(x1_0)], 1))
        else:
            x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        if self.use_deconv:
            x1_1 = self.conv1_1(torch.cat([x1_0, self.Up1_1(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up0_2(x1_1)], 1))
        else:
            x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        if self.use_deconv:
            x2_1 = self.conv2_1(torch.cat([x2_0, self.Up2_1(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up1_2(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up0_3(x1_2)], 1))
        else:
            x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))


        x4_0 = self.conv4_0(self.pool(x3_0))
        if self.use_deconv:
            x3_1 = self.conv3_1(torch.cat([x3_0, self.Up3_1(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up2_2(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up1_3(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up0_4(x1_3)], 1))
        else:
            x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))


        output = self.final(x0_4)
        return F.sigmoid(output)

class UNet_Nested2(nn.Module):

    def __init__(self, in_channels=6, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True,use_ae=False):
        super(UNet_Nested2, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.n_classes=n_classes
        self.use_ae=use_ae
        ae_out_channels=6

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        self.final_1_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        self.final_2_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        self.final_3_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        self.final_4_ae = nn.Conv2d(filters[0], ae_out_channels, 1)

        # self.final_conv=nn.Conv2d(4*n_classes,n_classes,1)
        # self.final_conv_ae=nn.Conv2d(4*ae_out_channels,ae_out_channels,1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def set_supervised(self,value):
        self.use_ae=not value

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        X_30 = self.conv30(maxpool2)  # 128*64*64
        maxpool3 = self.maxpool(X_30)  # 128*32*32
        X_40 = self.conv40(maxpool3)  # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        if self.use_ae:
            final_1 = self.final_1_ae(X_01)
            final_2 = self.final_2_ae(X_02)
            final_3 = self.final_3_ae(X_03)
            final_4 = self.final_4_ae(X_04)

            final = (final_1 + final_2 + final_3 + final_4) / 4

            return final
        else:
            # final layer
            final_1 = self.final_1(X_01)
            final_2 = self.final_2(X_02)
            final_3 = self.final_3(X_03)
            final_4 = self.final_4(X_04)

            final = (final_1 + final_2 + final_3 + final_4) / 4

            if self.n_classes == 1:
                if self.is_ds:
                    return F.sigmoid(final)
                else:
                    return F.sigmoid(final_4)
            else:
                if self.is_ds:
                    return final
                else:
                    return final_4
#=====================for unet_2D==============
class unet_2D(nn.Module):

    def __init__(self, feature_scale=2, n_classes=1, is_deconv=True, in_channels=6, is_batchnorm=True,use_dblock=False,dblock_type='ASPP'):
        super(unet_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        #self.use_dblock=use_dblock

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        #attention
        # self.up_concat4_att = unetUp_Att(filters[4], filters[3], self.is_deconv)
        # self.up_concat3_att = unetUp_Att(filters[3], filters[2], self.is_deconv)
        # self.up_concat2_att = unetUp_Att(filters[2], filters[1], self.is_deconv)
        # self.up_concat1_att = unetUp_Att(filters[1], filters[0], self.is_deconv)
        if dblock_type=='AS':
            logger.info("using AS for center...")
            self.use_dblock=True
            self.dblock=Dblock(filters[4])
        elif dblock_type=='ASPP':
            logger.info("using ASPP for center...")
            self.use_dblock = True
            self.dblock=ASPP(in_channel=filters[4])
        elif dblock_type=='DenseASPP':
            logger.info("using DenseASPP for center...")
            self.use_dblock = True
            self.dblock=DenseASPP(in_channel=filters[4])
        elif dblock_type=='FPA':
            logger.info("using FPA for center...")
            self.use_dblock = True
            self.dblock =FPAv2(filters[4],filters[4])
        else:
            logger.info("using no dblock for center...")
            self.use_dblock = False

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)#[1,6,256,256]==>[1,16,256,256]
        maxpool1 = self.maxpool1(conv1)#[1,16,256,256]==>[1,16,128,128]

        conv2 = self.conv2(maxpool1)#[1,16,128,128]==>[1,32,128,128]
        maxpool2 = self.maxpool2(conv2)#[1,32,128,128]==>[1,32,64,64]

        conv3 = self.conv3(maxpool2)#[1,32,64,64]==>[1,64,64,64]
        maxpool3 = self.maxpool3(conv3)#[1,64,64,64]==>[1,64,32,32]

        conv4 = self.conv4(maxpool3)#[1,64,32,32]==>[1,128,32,32]
        maxpool4 = self.maxpool4(conv4)#[1,128,32,32]==>[1,128,16,16]

        center = self.center(maxpool4)#[1,128,16,16]==>[1,256,16,16]
        if self.use_dblock:
            center=self.dblock(center)

        up4 = self.up_concat4(center,conv4)  # [1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
        up3 = self.up_concat3(up4,conv3)  # [1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
        up2 = self.up_concat2(up3,conv2)  # [1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
        up1 = self.up_concat1(up2,conv1)  # [1,16,256,256]+[1,32,128,128]+conv==>[1,16,256,256]

        final = self.final(up1)#[1,16,256,256]==>[1,1,256,256]
        if self.n_class==1:
            return F.sigmoid(final)# for binary only
        else:
            return final
    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
#===================SegNet 2D=================
class segUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2,use_att=False):
        super(segUp, self).__init__()
        #self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))


    def forward(self, high_feature):
        outputs0 = self.up(high_feature)
        # for feature in low_feature:
        #     outputs0 = torch.cat([outputs0, feature], 1)

        return outputs0
class SegNet_2D(nn.Module):

    def __init__(self, feature_scale=2, n_classes=1, is_deconv=True, in_channels=6, is_batchnorm=True,use_dblock=False,dblock_type='ASPP'):
        super(SegNet_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        #self.use_dblock=use_dblock

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = segUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = segUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = segUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = segUp(filters[1], filters[0], self.is_deconv)
        #attention
        # self.up_concat4_att = unetUp_Att(filters[4], filters[3], self.is_deconv)
        # self.up_concat3_att = unetUp_Att(filters[3], filters[2], self.is_deconv)
        # self.up_concat2_att = unetUp_Att(filters[2], filters[1], self.is_deconv)
        # self.up_concat1_att = unetUp_Att(filters[1], filters[0], self.is_deconv)
        if dblock_type=='AS':
            logger.info("using AS for center...")
            self.use_dblock=True
            self.dblock=Dblock(filters[4])
        elif dblock_type=='ASPP':
            logger.info("using ASPP for center...")
            self.use_dblock = True
            self.dblock=ASPP(in_channel=filters[4])
        elif dblock_type=='DenseASPP':
            logger.info("using DenseASPP for center...")
            self.use_dblock = True
            self.dblock=DenseASPP(in_channel=filters[4])
        elif dblock_type=='FPA':
            logger.info("using FPA for center...")
            self.use_dblock = True
            self.dblock =FPAv2(filters[4],filters[4])
        else:
            logger.info("using no dblock for center...")
            self.use_dblock = False

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)#[1,6,256,256]==>[1,16,256,256]
        maxpool1 = self.maxpool1(conv1)#[1,16,256,256]==>[1,16,128,128]

        conv2 = self.conv2(maxpool1)#[1,16,128,128]==>[1,32,128,128]
        maxpool2 = self.maxpool2(conv2)#[1,32,128,128]==>[1,32,64,64]

        conv3 = self.conv3(maxpool2)#[1,32,64,64]==>[1,64,64,64]
        maxpool3 = self.maxpool3(conv3)#[1,64,64,64]==>[1,64,32,32]

        conv4 = self.conv4(maxpool3)#[1,64,32,32]==>[1,128,32,32]
        maxpool4 = self.maxpool4(conv4)#[1,128,32,32]==>[1,128,16,16]

        center = self.center(maxpool4)#[1,128,16,16]==>[1,256,16,16]
        if self.use_dblock:
            center=self.dblock(center)

        up4 = self.up_concat4(center)  # [1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
        up3 = self.up_concat3(up4)  # [1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
        up2 = self.up_concat2(up3)  # [1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
        up1 = self.up_concat1(up2)  # [1,16,256,256]+[1,32,128,128]+conv==>[1,16,256,256]

        final = self.final(up1)#[1,16,256,256]==>[1,1,256,256]
        if self.n_class==1:
            return F.sigmoid(final)# for binary only
        else:
            return final
    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


