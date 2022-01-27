import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

import models.Satt_CD.modules.architecture as arch
import models.Satt_CD.modules.architecture_CD as arch_CD

import models.Satt_CD.modules.HRNet.model_zoo as modelZoo
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    '''
    kaiming初始化方法，论文在《 Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification》，
    公式推导同样从“方差一致性”出法，kaiming是针对xavier初始化方法在relu这一类激活函数表现不佳而提出的改进
    mode – either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass.
     Choosing 'fan_out' preserves the magnitudes in the backwards pass.
     nonlinearity – the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
    :param m:
    :param scale:
    :return:
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        if m.affine != False:

            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    ''''
    使得tensor是正交的，论文:Exact solutions to the nonlinear dynamics of learning in deep linear neural networks”
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)



def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':#在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0。推荐在ReLU网络中使用。
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':#主要用以解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用的参数初始化方法
        net.apply(weights_init_orthogonal)
    elif init_type == 'xavier':#Xavier初始化的基本思想是保持输入和输出的方差一致，这样就避免了所有输出值都趋向于0。这是通用的方法，适用于任何激活函数
        net.apply(weights_init_xavier)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt, device=None):
    gpu_ids = False
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    # in_channels = 6, n_classes = 1, filters = [], feature_scale = 4, is_deconv = True, is_batchnorm = True, use_res = True,
    # use_dense = True, use_deep_sub = True, att_type = 'CA', dblock_type = 'ASPP', use_rfnet = False

    if which_model == 'UNet_3PlusRes':
        netG = arch.UNet_3PlusRes(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'], filters=opt_net['filters'],
            feature_scale=opt_net['fea_scale'], is_deconv=opt_net['is_deconv'], is_batchnorm=opt_net['is_bn'],
                                use_res=opt_net['use_res'],use_dense=opt_net['use_dense'],use_deep_sup=opt_net['use_deep_sup'],
                                  att_type=opt_net['att_type'],dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'])
    elif which_model == 'UNet_3Plus':
        netG = arch.UNet_3Plus(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'], filters=opt_net['filters'],
            feature_scale=opt_net['fea_scale'], is_deconv=opt_net['is_deconv'], is_batchnorm=opt_net['is_bn'],
                                use_res=opt_net['use_res'],use_dense=opt_net['use_dense'],use_deep_sup=opt_net['use_deep_sup'],
                                  att_type=opt_net['att_type'],dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'])
    elif which_model == 'UNet_2D':
        netG=arch.unet_2D(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],use_res=opt_net['use_res'],
                          dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'])

    elif which_model == 'UNet_2D_Att':
        netG=arch.unet_2D_Att(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],use_res=opt_net['use_res'],
                          dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'],use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrain':
        netG=arch.unet_2D_PreTrain(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],use_res=opt_net['use_res'],
                          dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'],use_att=opt_net['use_att'],
                                   frozen_encoder=opt_net['frozen_encoder'],use_drop=opt_net["use_drop"],drop_rate=opt_net["drop_rate"])
    elif which_model == 'UNet_2D_Dense':
        netG=arch.unet_2D_Dense(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                          use_rfnet=opt_net['use_rfnet'],use_att=opt_net['use_att'],
                                   )
    elif which_model == 'UNet_2D_PreTrainED':
        netG=arch.unet_2D_PreTrainED(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],use_res=opt_net['use_res'],
                          dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'],use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrain_encoder_decoder':
        # netG = arch.unet_2D_PreTrainED(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
        #                                use_res=opt_net['use_res'],
        #                                dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
        #                                use_att=opt_net['use_att'])
        netEnc,netDec=arch.unet_2D_Encoder2(in_channels=opt_net['in_nc']),arch.unet_2D_Decoder2( n_classes=opt_net['out_nc'],use_rfnet=opt_net['use_rfnet'])
    elif which_model == 'UNet_2D_PreTrain256_ED2':
        netG = arch.unet_2D_PreTrain256_ED2(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                        use_res=opt_net['use_res'],
                                        dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                        use_att=opt_net['use_att'])

    elif which_model == 'UNet_2D_PreTrainED2':
        netG=arch.unet_2D_PreTrainED2(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],use_res=opt_net['use_res'],
                          dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'],use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrainC2':
        netG=arch.unet_2D_PreTrainC2(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],use_res=opt_net['use_res'],
                          dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'],use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrain_MS':
        netG = arch.unet_2D_PreTrain_MS(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                     use_res=opt_net['use_res'],
                                     dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                     use_att=opt_net['use_att'])

    elif which_model == 'UNet_2D_PreTrainScale':
        netG = arch.unet_2D_PreTrainScale(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                     use_res=opt_net['use_res'],
                                     dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                     use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrain_MSRef':
        netG = arch.unet_2D_PreTrain_MSRef(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                     use_res=opt_net['use_res'],
                                     dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                     use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrainDC':
        netG = arch.unet_2D_PreTrainDC(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                     use_res=opt_net['use_res'],
                                     dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                     use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrainSC':
        netG = arch. unet_2D_PreTrainSC(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                       use_res=opt_net['use_res'],
                                       dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                       use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrainSCRef':
        netG = arch.unet_2D_PreTrainSCRef(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                       use_res=opt_net['use_res'],
                                       dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                       use_att=opt_net['use_att'])
    elif which_model == 'ResNet_PreTrainSC':
        netG = arch.ResNet_PreTrainSC(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                          use_res=opt_net['use_res'],
                                          dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                          use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrainGFF':
        netG = arch.unet_2D_PreTrainGFF(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                          use_res=opt_net['use_res'],
                                          dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                          use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrainMC':
        netG = arch.unet_2D_PreTrainMC(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                          use_res=opt_net['use_res'],
                                          dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                          use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrainCCT':
        netG = arch.unet_2D_PreTrainCCT(opt['train'],in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'] )

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    # if opt['is_train']:
    #     init_weights(netG, init_type='kaiming', scale=0.1)#scale=0.1!

    #init_weights(netG, init_type='normal')
    #init_weights(netG, init_type='kaiming', scale=0.1)
    return netG





def define_G_CD(opt, device=None):
    gpu_ids = False
    opt_net = opt['network_G_CD']
    which_model = opt_net['which_model_G']
    '''
    self,in_c=3, use_att=False,nf=64,
                 att_mode='BAM'
    '''


    if which_model == 'Feat_Cmp':
        netG = arch_CD.FeatCmp_Net(in_c=opt_net["in_c"],nf=opt_net["nf"],use_att=opt_net["use_att"],att_mode=opt_net["att_mode"],
                                   backbone=opt_net["backbone"],drop_rate=opt_net["drop_rate"])
    elif which_model=='EDCls_Net':
        netG = arch_CD.EDCls_Net(in_c=opt_net["in_c"], nf=opt_net["nf"], use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"],backbone=opt_net["backbone"],out_c=opt_net["out_nc"])
    elif which_model == 'EDCls_Net2':
        netG = arch_CD.EDCls_Net2(in_c=opt_net["in_c"], nf=opt_net["nf"], use_att=opt_net["use_att"],
                                 att_mode=opt_net["att_mode"], backbone=opt_net["backbone"], out_c=opt_net["out_nc"],
                                  drop_rate=opt_net["drop_rate"])

    elif which_model=='EDCls_UNet':

        net_feat=arch.unet_2D_PreTrainInter(in_channels=opt_net["in_c"])
        netG = arch_CD.EDCls_UNet(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],net_feat=net_feat)

    elif which_model=='EDCls_UNet2':

        net_feat=arch.unet_2D_Encoder(in_channels=opt_net["in_c"],use_se=False,dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet2(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],net_feat=net_feat,use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"],diff_mode=opt_net["diff_mode"],use_DS=opt_net["use_DS"],drop_rate=opt["train"]["drop_rate"],use_CatOut=opt["train"]["use_CatOut"],
                                   use_se=opt_net["use_se"],training_mode=opt_net["training_mode"])









    elif which_model=='EDCls_UNet2_New2':#for SCDNet in_c=3+3, out_c=7+7

        net_feat=arch.unet_2D_Encoder(in_channels=opt_net["in_c"],use_se=False,dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet2_New2(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],net_feat=net_feat,use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"],diff_mode=opt_net["diff_mode"],use_DS=opt_net["use_DS"],
                                       drop_rate=opt["train"]["drop_rate"],use_CatOut=opt["train"]["use_CatOut"],
                                        use_se=opt_net["use_se"],se_block=opt_net["se_block"],training_mode=opt_net["training_mode"])
    elif which_model=='EDCls_UNet_MC7Bin':#for SCDNet in_c=3+3, out_c=7+7

        net_feat=arch.unet_2D_Encoder(in_channels=opt_net["in_c"],use_se=False,dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet2_MC7Bin(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],net_feat=net_feat,use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"],diff_mode=opt_net["diff_mode"],use_DS=opt_net["use_DS"],
                                       drop_rate=opt["train"]["drop_rate"],use_CatOut=opt["train"]["use_CatOut"],
                                        use_se=opt_net["use_se"],se_block=opt_net["se_block"],training_mode=opt_net["training_mode"])

    elif which_model=='EDCls_UNet_MC6Bin':#for SCDNet in_c=3+3, out_c=7+7

        net_feat=arch.unet_2D_Encoder(in_channels=opt_net["in_c"],use_se=False,dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet2_MC6Bin(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],net_feat=net_feat,use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"],diff_mode=opt_net["diff_mode"],use_DS=opt_net["use_DS"],
                                       drop_rate=opt["train"]["drop_rate"],use_CatOut=opt["train"]["use_CatOut"],
                                        use_se=opt_net["use_se"],se_block=opt_net["se_block"],training_mode=opt_net["training_mode"])

    elif which_model=='DeepLab_SCD':
        netG = DL.DeepLab_SCD(backbone='drn', num_classes=opt_net['out_nc'],ASPP_type=opt_net['ASPP_type'])
    elif which_model=='HRNet_SCD':
        #netG = DL.DeepLab_SCD(backbone='drn', num_classes=opt_net['out_nc'],ASPP_type=opt_net['ASPP_type'])
        #self.model = get_model(args.model, args.backbone, args.pretrained,len(trainset.CLASSES)-1, args.lightweight)
        netG=modelZoo.get_model(opt_net['model_head'],opt_net['backbone'],opt_net['pretrained'],opt_net['out_nc'],opt_net['lightweight'])

    elif which_model=='EDCls_UNet3':#for ISCD in_c=3+3, out_c=7+7
        net_feat = arch.unet_2D_Encoder(in_channels=opt_net["in_c"], use_se=False, dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet3(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],
                                       net_feat=net_feat, use_att=opt_net["use_att"],
                                       att_mode=opt_net["att_mode"], diff_mode=opt_net["diff_mode"],
                                       use_DS=opt_net["use_DS"],
                                       drop_rate=opt["train"]["drop_rate"])

    elif which_model == 'EDCls_UNet_BCD_WHU_Flow':#for SSCD in_c=3+3,out_c=1

        net_feat = arch.unet_2D_Encoder(in_channels=opt_net["in_c"], use_se=False, dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet_BCD_WHU_Flow(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"], net_feat=net_feat,
                                   use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"], diff_mode=opt_net["diff_mode"],
                                   use_DS=opt_net["use_DS"], drop_rate=opt["train"]["drop_rate"],
                                   use_CatOut=opt["train"]["use_CatOut"],
                                   use_se=opt_net["use_se"])




    elif which_model=='EDCls_UNet2_Res':

        net_feat=arch.unet_2D_Encoder(in_channels=opt_net["in_c"],use_se=False,dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet2_Res(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],net_feat=net_feat,use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"],diff_mode=opt_net["diff_mode"],use_DS=opt_net["use_DS"],
                                       drop_rate=opt["train"]["drop_rate"],use_CatOut=opt["train"]["use_CatOut"],use_se=opt_net["use_se"],se_block=opt_net["se_block"],training_mode=opt_net["training_mode"])
    elif which_model=='EDCls_UNet3_Plus':
        net_feat = arch.unet_2D_Encoder(in_channels=opt_net["in_c"], use_se=False, dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet3_Plus(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],
                                       net_feat=net_feat, use_att=opt_net["use_att"],
                                       att_mode=opt_net["att_mode"], diff_mode=opt_net["diff_mode"],
                                       use_DS=opt_net["use_DS"],
                                       drop_rate=opt["train"]["drop_rate"], use_CatOut=opt["train"]["use_CatOut"],
                                       use_se=opt["train"]["use_se"])

    elif which_model=='EDCls_UNet2_New5':

        net_feat=arch.unet_2D_Encoder_New(in_channels=opt_net["in_c"],use_se=opt["train"]["use_se"],dblock_type=opt_net['dblock_type'])
        netG = arch_CD.EDCls_UNet2_New5(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],net_feat=net_feat,use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"],diff_mode=opt_net["diff_mode"],use_DS=opt_net["use_DS"],drop_rate=opt["train"]["drop_rate"],use_CatOut=opt["train"]["use_CatOut"],use_se=opt["train"]["use_se"])

    elif which_model=='EDCls_UNet2_Res50':

        net_feat=arch.unet_2D_Encoder_Res50(in_channels=opt_net["in_c"],use_se=opt["train"]["use_se"])
        netG = arch_CD.EDCls_UNet2_Res50(in_c=opt_net["in_c"], nf=opt_net["nf"], out_c=opt_net["out_nc"],net_feat=net_feat,use_att=opt_net["use_att"],
                                   att_mode=opt_net["att_mode"],diff_mode=opt_net["diff_mode"],use_DS=opt_net["use_DS"],drop_rate=opt["train"]["drop_rate"],use_CatOut=opt["train"]["use_CatOut"],use_se=opt["train"]["use_se"])

    elif which_model == 'FC_EF':
        netG=arch_CD.unet_2D(dblock_type = opt_net['dblock_type'])
    elif which_model == 'Seg_EF':
        netG=arch_CD.SegNet_2D(dblock_type = opt_net['dblock_type'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))




    return netG

# Discriminator
def define_D(opt):

    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model=='discriminator_fc_256':
        netD=arch.Discriminator_FC_256(opt_net['in_nc'],act=False)
    elif which_model=='discriminator_fc_pix':
        netD=arch.Discriminator_FC_Pix(opt_net["in_nc"])
    elif which_model=='discriminator_fc_512':
        netD=arch.Discriminator_FC_512(opt_net['in_nc'],act=False)
    elif which_model == 'discriminator_vgg_256':
        netD = arch.Discriminator_VGG_256(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    #init_weights(netD, init_type='kaiming', scale=1)
    init_weights(netD,init_type='normal')


    return netD







def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)

    netF.eval()
    return netF
