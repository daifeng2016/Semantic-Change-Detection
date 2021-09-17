import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
#import kornia
from . import networks as networks
from .base_model import BaseModel
from models.Satt_CD.modules.loss import GANLoss, GradientPenaltyLoss,ComboLoss,abCE_loss
from models.utils import one_hot_cuda
logger = logging.getLogger('base')

import torch.nn.functional as F
import pdb
from losses.myLoss import bce_edge_loss
from torch.autograd import Variable
import numpy as np
from math import exp
from models.Satt_CD.modules import block as B
from models.Satt_CD.modules.loss import BCL,abCE_loss
from models.Satt_CD.modules.adamw import AdamW
from losses.myLoss import N8ASCLoss


class CD_Model(BaseModel):
    def __init__(self, opt):
        super(CD_Model, self).__init__(opt)
        train_opt = opt['train']
        # self.device='cpu'
        #=============define networks and load pretrained models==================
        # if train_opt["use_encoder_decoder"]:
        #     self.netEnc,self.netDec= networks.define_G(opt).to(self.device)  # G
        # else:
        #self.netG = networks.define_G(opt).to(self.device)  # G (can has encoder, main_decoder,aux_decoder if SSL setting )
        self.netG=networks.define_G_CD(opt).to(self.device)
        if opt["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_Seg":
            self.netCD=networks.define_G_CD0(opt).to(self.device)
        #====================for deeplab========================

        self.netD_seg=None
        self.netD_fea=None
        self.netG_AdaIN=None
        if train_opt['is_adv_train']:
            self.netD_seg = networks.define_D(opt).to(self.device)  # D
            self.netD_fea = networks.define_D_grad(opt).to(self.device)  # D_grad

            #self.netD_MS=networks.define_D_MS(opt).to(self.device)

            self.netD_fea.train()
            self.netD_seg.train()
            #self.netD_MS.train()


        self.cri_seg_loss=bce_edge_loss(use_edge=True).to(self.device)
        self.cri_seg0_loss=bce_edge_loss(use_edge=False).to(self.device)
        self.cri_dist=BCL().to(self.device)
        self.cri_seg_mc=ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        class_weight=torch.FloatTensor(train_opt["class_weight"])
        # if train_opt["use_label_smooth"]:
        #     logger.info("using label smooth...")
        #     from models.Satt_CD.modules.loss import LabelSmoothLoss
        #     self.cri_ce_loss=LabelSmoothLoss(smoothing=0.1).to(self.device)
        #
        # else:
        #     if train_opt["use_MC6"]:
        #         weight = torch.FloatTensor([2, 1, 2, 2, 1, 1]).cuda()  # for MC6
        #         self.cri_ce_loss = nn.CrossEntropyLoss(ignore_index=-1, weight=weight).to(self.device)
        #     else:
        #
        #         self.cri_ce_loss = nn.CrossEntropyLoss().to(self.device)
        if opt["network_G_CD"]["which_model_G"] == "HRNet_SCD":
            opt["network_G_CD"]["use_DS"]=False
            if opt["dataset_name"]=="sensetime":
                weight = torch.FloatTensor([2, 1, 2, 2, 1, 1]).cuda()  # for MC6
            else:
                weight = torch.FloatTensor([1, 2, 1, 2, 1]).cuda()  # for MC6
            self.cri_ce_loss = nn.CrossEntropyLoss(ignore_index=-1, weight=weight).to(self.device)
        else:

            self.cri_ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.use_abCE=False

        self.ce_weight=train_opt['ce_weight']
        self.use_NEloss=False
        #self.cri_seg_loss = nn.BCELoss().to(self.device)
        self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1  # 1
        self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0  # 0
        self.G_update_ratio = train_opt['G_update_ratio'] if train_opt['G_update_ratio'] else 2  # 1
        # optimizers
        #===================================G=================================
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params = []
        message = "lr is {:.3e}".format(train_opt['lr_G'])
        logger.info(message)
        message="wd is {:.3e}".format(wd_G)
        logger.info(message)
        if self.netG:
            for k, v in self.netG.named_parameters():  # optimize part of the model

                if v.requires_grad:
                    optim_params.append(v)  # list, without name
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            # self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
            #                                     weight_decay=wd_G, betas=(
            #         train_opt['beta1_G'], 0.999))  # for Adam, no need to use weight_decay
            # self.optimizer_G = torch.optim.SGD(optim_params,
            #                       lr=train_opt['lr_G'], momentum=train_opt["momentum"], weight_decay=wd_G)#wd_G set to 1e-6

            self.optimizer_G = AdamW(optim_params, lr=train_opt['lr_G'],weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)  # self.netG.parameters()


        #################################################################################################
        #===================================D============================================================
        #################################################################################################
        if train_opt['is_adv_train']:
            # GD gan loss
           self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
           self.l_gan_w = train_opt['gan_weight']
           wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
           self.optimizer_D_fea = torch.optim.Adam(self.netD_fea.parameters(), lr=train_opt['lr_D'], \
                                            weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
           self.optimizer_D_seg = torch.optim.Adam(self.netD_seg.parameters(), lr=train_opt['lr_D'], \
                                               weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
           # self.optimizer_D_MS = torch.optim.Adam(self.netD_MS.parameters(), lr=train_opt['lr_D'], \
           #                                         weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))

           self.optimizers.append(self.optimizer_D_fea)
           self.optimizers.append(self.optimizer_D_seg)
           #self.optimizers.append(self.optimizer_D_MS)

        #================================schedulers==========================
        if train_opt['lr_scheme'] == 'MultiStepLR':
            if train_opt["warmup_epoch"]>0:


                from torch_warmup_lr import WarmupLR


                for optimizer in self.optimizers:


                    my_scheduler=lr_scheduler.MultiStepLR(optimizer, \
                                                                    milestones=[5,10,15], gamma=train_opt['lr_gamma'])
                    my_scheduler=WarmupLR(my_scheduler, init_lr=train_opt['lr_G']*0.05 , num_warmup=train_opt["warmup_epoch"], warmup_strategy='cos')

                    self.schedulers.append(my_scheduler)


            else:
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                                                                    train_opt['lr_steps'], train_opt['lr_gamma']))


        elif train_opt['lr_scheme'] == 'CosineLR':
            if train_opt["warmup_epoch"]>0:


                from torch_warmup_lr import WarmupLR


                for optimizer in self.optimizers:


                    my_scheduler=lr_scheduler.CosineAnnealingLR(optimizer, \
                                                                          T_max=train_opt["nepoch"] - train_opt["warmup_epoch"])

                    my_scheduler=WarmupLR(my_scheduler, init_lr=train_opt['lr_G']*0.05 , num_warmup=train_opt["warmup_epoch"], warmup_strategy='cos')

                    self.schedulers.append(my_scheduler)


            else:
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, \
                                                                          T_max=train_opt["nepoch"]))
        else:

            logger.info("using cos-ensemble with consine lr")
            #raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        self.log_dict = OrderedDict()
        self.total_iters=train_opt['niter']
        #self.PREHEAT_STEPS=int(self.total_iters/3)#===for warmup
        self.PREHEAT_STEPS=0
        #=======Labels for Adversarial Training===========
        source_label = 1
        target_label = 0
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.epsion=train_opt['epsion']
        self.config = opt
        self.ita=train_opt['ita']
        self.train_opt=train_opt
        self.ds=self.config.ds

    def feed_data(self, data):

        self.img,self.label=data["img"],data["label"]
        self.img,self.label=self.img.to(self.device),self.label.to(self.device)
        if self.config["mode"]=='Train' and self.config["network_G_CD"]["which_model_G"]=='Feat_Cmp':

            self.label = F.interpolate(self.label, size=torch.Size([self.img.shape[2]//self.ds, self.img.shape[3]//self.ds]),mode='nearest')
            self.label[self.label == 1] = -1  # change
            self.label[self.label == 0] = 1  # no change

    def feed_data_batch_st(self, batch_s,bacht_t):

        self.batch_s=batch_s
        self.batch_t=bacht_t

















    def optimize_parameters(self, step):
        #=============G==================
        self.optimizer_G.zero_grad()
        l_g_total=0
        images_T12=self.img
        labels=self.label

        images_T1=images_T12[:,0:3,...]
        images_T2=images_T12[:,3:6,...]

        featT1,featT2=self.netG(images_T1,images_T2)
        dist = F.pairwise_distance(featT1,featT2, keepdim=True)
        dist = F.interpolate(dist, size=images_T1.shape[2:], mode='bilinear', align_corners=True)

        #self.preds=(dist > 1).float()



        l_g_total = self.cri_dist(dist, labels)
        l_g_total.backward()
        self.optimizer_G.step()


        # set log
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = 0

    def optimize_parameters_MC(self, step):#for DSCD
        #============direct SCD=================

        self.optimizer_G.zero_grad()
        l_g_total = 0
        images_T12 = self.img
        labels = self.label
        class_num = self.config["network_G_CD"]["out_nc"]
        one_hot_labels=one_hot_cuda(labels,num_classes=class_num)

        images_T1 = images_T12[:, 0:3, ...]
        images_T2 = images_T12[:, 3:6, ...]

        pred = self.netG(images_T12)
        if self.config.dataset_name=='sensetime':
           class_rate=np.array([0.800605,
0.001078,
0.001159,
0.000218,
0.000292,
0.000011,
0.001013,
0.024490,
0.009264,
0.049642,
0.001129,
0.001250,
0.032402,
0.006272,
0.024567,
0.000400,
0.000122,
0.007774,
0.003105,
0.003033,
0.000031,
0.000108,
0.016380,
0.005004,
0.001354,
0.008437,
0.000107,
0.000003,
0.000480,
0.000117,
0.000031,
0.000124
])#for sensetime dataset
        else:
           class_rate=np.array([0.99232,
                      0.00011,0.00011,0.00011,
                      0.00653,0.00001,0.00077,
                      0.00014,0.00002,
                      0.00001,0.00004,0.00004])#for HRSCD dataset

        class_weight=(1-class_rate)/(class_num-1)
        for k in range (class_num):
            l_g_total+=class_weight[k]*self.cri_seg_mc(pred[:,k,...],one_hot_labels[:,k,...])

        ce_weight=self.config["train"]["ce_weight"]

        l_g_total+=self.cri_ce_loss(pred,labels)*ce_weight




        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = 0

    def optimize_parameters_MC7_DS(self, step):
        # =============G==================
        self.optimizer_G.zero_grad()
        l_g_total = 0
        images_T12 = self.img
        labels = self.label
        class_num = self.config["network_G_CD"]["out_nc"]
        # one_hot_labels = one_hot_cuda(labels, num_classes=class_num)
        label_smooth = self.config["train"]["use_label_smooth"]
        one_hot_labels1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
        one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

        images_T1 = images_T12[:, 0:3, ...]
        images_T2 = images_T12[:, 3:6, ...]


        class_rate1 = np.array([0.800605, 0.0028, 0.0855, 0.0649, 0.0141, 0.0314, 0.000755])  # 1==>
        class_rate2 = np.array([0.800605, 0.002496, 0.058114, 0.033875, 0.017139, 0.086095, 0.001678])  # ==>1
        # class_weight1 = (1 - class_rate1) / (class_num - 1)
        # class_weight2 = (1 - class_rate2) / (class_num - 1)
        class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
        class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]

        if self.config["train"]["use_DS"]:

            pred1, pred2, pred3, pred4,pred5 = self.netG(images_T1, images_T2)
            if self.config.patch_size == 256:
                img_down_size = 16
            else:
                img_down_size = 32

            labels1 = F.interpolate(labels.float(), (img_down_size, img_down_size), mode='bilinear', align_corners=True)
            labels2 = F.interpolate(labels.float(), (img_down_size * 2, img_down_size * 2), mode='bilinear',
                                    align_corners=True)
            labels3 = F.interpolate(labels.float(), (img_down_size * 4, img_down_size * 4), mode='bilinear',
                                    align_corners=True)
            labels4 = F.interpolate(labels.float(), (img_down_size * 8, img_down_size * 8), mode='bilinear',
                                    align_corners=True)
            labels1, labels2, labels3,labels4 = labels1.long(), labels2.long(), labels3.long(),labels4.long()


            one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_1 = one_hot_cuda(labels4[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_2 = one_hot_cuda(labels4[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels4_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels4_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred5[0][:, k, ...], one_hot_labels1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred5[1][:, k, ...], one_hot_labels2[:, k, ...])

            ce_weight = 4.0

            l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred1[1],
                                                                                                       labels1[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2[1],
                                                                                                       labels2[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred3[1],
                                                                                                       labels3[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred4[0], labels4[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred4[1],
                                                                                                      labels4[:, 1,
                                                                                                      ...]) * ce_weight

            l_g_total += self.cri_ce_loss(pred5[0], labels[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred5[1],
                                                                                                      labels[:, 1,
                                                                                                      ...]) * ce_weight

            # ======================for FocalLossWithDice============
            # ce_weight=2.0
            # d_weight=0.5
            # # class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
            # # class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
            # weight=[0.01,3,1,1,1,1,9]
            # from models.Satt_CD.modules.loss import FocalLossWithDice
            # myLoss=FocalLossWithDice(ce_weight=ce_weight,d_weight=d_weight,weight=weight)
            # l_g_total+=myLoss(pred1[0],labels1[:,0,...])+myLoss(pred1[1],labels1[:,1,...])
            # l_g_total += myLoss(pred2[0], labels2[:, 0, ...]) + myLoss(pred2[1], labels2[:, 1, ...])
            # l_g_total += myLoss(pred3[0], labels3[:, 0, ...]) + myLoss(pred3[1], labels3[:, 1, ...])
            # l_g_total += myLoss(pred4[0], labels[:, 0, ...]) + myLoss(pred4[1], labels[:, 1, ...])
            # ================================================




        else:
            pred1, pred2 = self.netG(images_T1, images_T2)
            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
            l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * 10 + self.cri_ce_loss(pred2,
                                                                                            labels[:, 1, ...]) * 10

        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = 0


    def optimize_parameters_MC7_bin(self, step):
        # =============G==================
        self.optimizer_G.zero_grad()
        l_g_total = 0
        images_T12 = self.img
        labels = self.label
        class_num = self.config["network_G_CD"]["out_nc"]
        if self.config["train"]["use_MC6"]:
            labels-=1

        label_smooth = False
        one_hot_labels1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
        one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

        images_T1 = images_T12[:, 0:3, ...]
        images_T2 = images_T12[:, 3:6, ...]

        # class_rate = np.array([0.800605,
        #                                  0.001078,0.001159,0.000218,0.000292,0.000011,#1
        #                        0.001013,          0.024490,0.009264,0.049642,0.001129,
        #                        0.001250,0.032402,          0.006272,0.024567,0.000400,#3
        #                        0.000122,0.007774,0.003105,          0.003033,0.000031,
        #                        0.000108, 0.016380,0.005004,0.001354,0.008437,0.000107,#5
        #                        0.000003,0.000480,0.000117,0.000031,0.000124#6
        #                        ])
        # class_rate1=np.array([0.800605,0.0028,0.0855, 0.0649, 0.0141,0.0314,0.000755])#1==>
        # class_rate2=np.array([0.800605,0.002496, 0.058114, 0.033875, 0.017139, 0.086095, 0.001678])#==>1
        # class_weight1 = (1 - class_rate1) / (class_num - 1)
        # class_weight2 = (1 - class_rate2) / (class_num - 1)
        # =====================for sensetime cd==================
        if self.config["dataset_name"] == 'sensetime':
            class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
            class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]

            # class_weight1=[0.1972,    0.0065,    0.0085,    0.0392,    0.0176,    0.7312]
            # class_weight2=[0.3599,    0.0155,    0.0265,    0.0524,    0.0104,    0.5353]

        else:
            class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
            class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]
        # ======================for franch cd====================
        # class_rate1 = np.array([0.99232, 0.00013, 0.00731, 0.00016, 0, 0.00009])  # 1==>
        # class_rate2 = np.array([0.99232, 0.000803, 0.00017, 0.00001, 0.00005, 0.00078])  # ==>1
        # class_weight1 = [0.0001,    0.3054,    0.0054,    0.2481, 0,    0.4410]
        # class_weight2 = [0.0001,    0.0097,    0.0458,    0.7787,    0.1557,    0.0100]

        if self.config["network_G_CD"]["use_DS"]:

            pred1, pred2, pred3, pred4 = self.netG(images_T1, images_T2)
            if self.config.patch_size == 256:
                img_down_size = 16
            else:
                img_down_size = 32
            if self.config["train"]["use_progressive_resize"]:
                img_down_size = 16

            labels1 = F.interpolate(labels.float(), (img_down_size, img_down_size), mode='bilinear', align_corners=True)
            labels2 = F.interpolate(labels.float(), (img_down_size * 2, img_down_size * 2), mode='bilinear',
                                    align_corners=True)
            labels3 = F.interpolate(labels.float(), (img_down_size * 4, img_down_size * 4), mode='bilinear',
                                    align_corners=True)
            labels1, labels2, labels3 = labels1.long(), labels2.long(), labels3.long()

            one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

            # for k in range(class_num):
            #     l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
            #     l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])
            #
            #     l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
            #     l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])
            #
            #     l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
            #     l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])
            #
            #     l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels1[:, k, ...])
            #     l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels2[:, k, ...])

            # if torch.isnan(l_g_total):
            #     import cv2
            #     for i in range(images_T1.shape[0]):
            #         img1 = images_T1[i].data.cpu().numpy().transpose(1, 2, 0) * 255
            #         img2 = images_T2[i].data.cpu().numpy().transpose(1, 2, 0) * 255
            #         label1 = labels[i, 0, ...].data.cpu().numpy()
            #         label2 = labels[i, 1, ...].data.cpu().numpy()
            #         # cv2.imwrite((train_dir + '/img/%d.jpg' % (len(train_img) + i)), trainImg[i])
            #         cv2.imwrite(self.config.log_dir + '/error/img1_%d.png' % i, img1.astype('uint8'))
            #         cv2.imwrite(self.config.log_dir + '/error/img2_%d.png' % i, img2.astype('uint8'))
            #         cv2.imwrite(self.config.log_dir + '/error/label1_%d.png' % i, label1.astype('uint8'))
            #         cv2.imwrite(self.config.log_dir + '/error/label2_%d.png' % i, label2.astype('uint8'))
            #     raise ValueError('model loss is  none')

            ce_weight = self.ce_weight
            if self.use_abCE:
                l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...],
                                              curr_iter=step) * ce_weight + self.cri_ce_loss(pred1[1],
                                                                                             labels1[:, 1,
                                                                                             ...],
                                                                                             curr_iter=step) * ce_weight
                l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...],
                                              curr_iter=step) * ce_weight + self.cri_ce_loss(pred2[1],
                                                                                             labels2[:, 1,
                                                                                             ...],
                                                                                             curr_iter=step) * ce_weight
                l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...],
                                              curr_iter=step) * ce_weight + self.cri_ce_loss(pred3[1],
                                                                                             labels3[:, 1,
                                                                                             ...],
                                                                                             curr_iter=step) * ce_weight
                l_g_total += self.cri_ce_loss(pred4[0], labels[:, 0, ...],
                                              curr_iter=step) * ce_weight + self.cri_ce_loss(pred4[1],
                                                                                             labels[:, 1,
                                                                                             ...],
                                                                                             curr_iter=step) * ce_weight
            else:
                l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred1[1],
                                                                                                           labels1[:, 1,
                                                                                                           ...]) * ce_weight
                l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2[1],
                                                                                                           labels2[:, 1,
                                                                                                           ...]) * ce_weight
                l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred3[1],
                                                                                                           labels3[:, 1,
                                                                                                           ...]) * ce_weight
                l_g_total += self.cri_ce_loss(pred4[0], labels[:, 0, ...]) * ce_weight * 2 + self.cri_ce_loss(pred4[1],
                                                                                                              labels[:,
                                                                                                              1,
                                                                                                              ...]) * ce_weight * 2
            if self.use_NEloss:
                l_g_NE = 0.01 * (N8ASCLoss(pred1[0]) + N8ASCLoss(pred1[1]) + \
                                 N8ASCLoss(pred2[0]) + N8ASCLoss(pred2[1]) + \
                                 N8ASCLoss(pred3[0]) + N8ASCLoss(pred3[1])) + \
                         0.05 * (N8ASCLoss(pred4[0]) + N8ASCLoss(pred4[1]))
                l_g_total += l_g_NE

            # ===ensure the changed and unchanged part are the same
            labels1_f12 = torch.zeros_like(labels1[:, 0, ...])
            labels1_f12[labels1[:, 0, ...] > 0] = 1.0
            labels2_f12 = torch.zeros_like(labels2[:, 0, ...])
            labels2_f12[labels2[:, 0, ...] > 0] = 1.0
            labels3_f12 = torch.zeros_like(labels3[:, 0, ...])
            labels3_f12[labels3[:, 0, ...] > 0] = 1.0
            labels4_f12 = torch.zeros_like(labels[:, 0, ...])
            labels4_f12[labels[:, 0, ...] > 0] = 1.0
            labels1_f12 = labels1_f12.unsqueeze(1).float()
            labels2_f12 = labels2_f12.unsqueeze(1).float()
            labels3_f12 = labels3_f12.unsqueeze(1).float()
            labels4_f12 = labels4_f12.unsqueeze(1).float()
            bce_loss = nn.BCELoss(reduction='none')
            loss1=bce_loss(pred1[2],labels1_f12)
            loss1[labels1_f12 == 1] *= 2
            loss1 = loss1.mean()

            loss2 = bce_loss(pred2[2], labels2_f12)
            loss2[labels2_f12 == 1] *= 2
            loss2 = loss2.mean()

            loss3 = bce_loss(pred3[2], labels3_f12)
            loss3[labels3_f12 == 1] *= 2
            loss3 = loss3.mean()

            loss4 = bce_loss(pred4[2], labels4_f12)
            loss4[labels4_f12 == 1] *= 2
            loss4 = loss4.mean()

            l_g_total+=loss1+loss2+loss3+2*loss4












        else:
            pred1, pred2 ,pred12= self.netG(images_T1, images_T2)
            ce_weight = self.ce_weight
            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
            l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * ce_weight*2 + self.cri_ce_loss(pred2,
                                                                                            labels[:, 1, ...]) * ce_weight*2
            labels4_f12 = torch.zeros_like(labels[:, 0, ...])
            labels4_f12[labels[:, 0, ...] > 0] = 1.0
            labels4_f12 = labels4_f12.unsqueeze(1).float()
            bce_loss = nn.BCELoss(reduction='none')
            loss4 = bce_loss(pred12, labels4_f12)
            loss4[labels4_f12 == 1] *= 2
            loss4 = loss4.mean()

            l_g_total += 2 * loss4



        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = 0


    def optimize_parameters_MC6_bin(self, step):
        # =============G==================
        self.optimizer_G.zero_grad()
        l_g_total = 0
        images_T12 = self.img
        labels = self.label
        class_num = self.config["network_G_CD"]["out_nc"]
        if self.config["train"]["use_MC6"]:
            labels-=1

        label_smooth = False
        one_hot_labels1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
        one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

        images_T1 = images_T12[:, 0:3, ...]
        images_T2 = images_T12[:, 3:6, ...]

        # class_rate = np.array([0.800605,
        #                                  0.001078,0.001159,0.000218,0.000292,0.000011,#1
        #                        0.001013,          0.024490,0.009264,0.049642,0.001129,
        #                        0.001250,0.032402,          0.006272,0.024567,0.000400,#3
        #                        0.000122,0.007774,0.003105,          0.003033,0.000031,
        #                        0.000108, 0.016380,0.005004,0.001354,0.008437,0.000107,#5
        #                        0.000003,0.000480,0.000117,0.000031,0.000124#6
        #                        ])
        # class_rate1=np.array([0.800605,0.0028,0.0855, 0.0649, 0.0141,0.0314,0.000755])#1==>
        # class_rate2=np.array([0.800605,0.002496, 0.058114, 0.033875, 0.017139, 0.086095, 0.001678])#==>1
        # class_weight1 = (1 - class_rate1) / (class_num - 1)
        # class_weight2 = (1 - class_rate2) / (class_num - 1)
        # =====================for sensetime cd==================
        if self.config["dataset_name"] == 'sensetime':
            class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
            class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]

            # class_weight1=[0.1972,    0.0065,    0.0085,    0.0392,    0.0176,    0.7312]
            # class_weight2=[0.3599,    0.0155,    0.0265,    0.0524,    0.0104,    0.5353]

        else:
            class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
            class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]

        pred1, pred2, pred12 = self.netG(images_T1, images_T2)
        ce_weight = self.ce_weight
        # for k in range(class_num):
        #     l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
        #     l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
        l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) + self.cri_ce_loss(pred2,labels[:, 1, ...])

        labels4_f12 = torch.zeros_like(labels[:, 0, ...])
        labels4_f12[labels[:, 0, ...] >= 0] = 1.0
        labels4_f12 = labels4_f12.unsqueeze(1).float()
        bce_loss = nn.BCELoss(reduction='none')
        loss4 = bce_loss(pred12, labels4_f12)
        loss4[labels4_f12 == 1] *= 2
        loss4 = loss4.mean()

        l_g_total += 2 * loss4



        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = 0



    def optimize_parameters_MC7(self, step):
        # =============G==================
        self.optimizer_G.zero_grad()
        l_g_total = 0
        images_T12 = self.img
        labels = self.label
        class_num = self.config["network_G_CD"]["out_nc"]
        #one_hot_labels = one_hot_cuda(labels, num_classes=class_num)
        label_smooth = self.config["train"]["use_label_smooth"]
        one_hot_labels1=one_hot_cuda(labels[:,0,...],num_classes=class_num,label_smooth=label_smooth)
        one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num,label_smooth=label_smooth)

        images_T1 = images_T12[:, 0:3, ...]
        images_T2 = images_T12[:, 3:6, ...]


        # class_rate = np.array([0.800605,
        #                                  0.001078,0.001159,0.000218,0.000292,0.000011,#1
        #                        0.001013,          0.024490,0.009264,0.049642,0.001129,
        #                        0.001250,0.032402,          0.006272,0.024567,0.000400,#3
        #                        0.000122,0.007774,0.003105,          0.003033,0.000031,
        #                        0.000108, 0.016380,0.005004,0.001354,0.008437,0.000107,#5
        #                        0.000003,0.000480,0.000117,0.000031,0.000124#6
        #                        ])
        # class_rate1=np.array([0.800605,0.0028,0.0855, 0.0649, 0.0141,0.0314,0.000755])#1==>
        # class_rate2=np.array([0.800605,0.002496, 0.058114, 0.033875, 0.017139, 0.086095, 0.001678])#==>1
        # class_weight1 = (1 - class_rate1) / (class_num - 1)
        # class_weight2 = (1 - class_rate2) / (class_num - 1)
        #===============================for sensetime cd======================================

        #======================for franch cd====================
        # class_rate1 = np.array([0.99232, 0.00013, 0.00731, 0.00016, 0, 0.00009])  # 1==>
        # class_rate2 = np.array([0.99232, 0.000803, 0.00017, 0.00001, 0.00005, 0.00078])  # ==>1
        #===================================================================================

        if self.config["dataset_name"]=='sensetime':
            class_weight1 = [ 0.0007,    0.1970,    0.0065,    0.0085,    0.0391,    0.0176,    0.7307]
            class_weight2 = [0.0011,    0.3595,    0.0154,    0.0265,    0.0524,    0.0104,    0.5347]
        else:
            class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
            class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]


        if self.config["network_G_CD"]["use_DS"]:

            pred1,pred2,pred3,pred4 = self.netG(images_T1, images_T2)
            if self.config.patch_size==256:
                img_down_size=16
            else:
                img_down_size = 32
            if self.config["train"]["use_progressive_resize"]:
                img_down_size=16

            labels1=F.interpolate(labels.float(),(img_down_size,img_down_size),mode='bilinear',align_corners=True)
            labels2= F.interpolate(labels.float(), (img_down_size*2, img_down_size*2), mode='bilinear', align_corners=True)
            labels3= F.interpolate(labels.float(), (img_down_size*4, img_down_size*4), mode='bilinear', align_corners=True)
            labels1,labels2,labels3=labels1.long(),labels2.long(),labels3.long()


            one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num,label_smooth=label_smooth)
            one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num,label_smooth=label_smooth)
            one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num,label_smooth=label_smooth)
            one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num,label_smooth=label_smooth)
            one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num,label_smooth=label_smooth)
            one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num,label_smooth=label_smooth)

            #========================================one_hot_loss===============================================
            if self.config["train"]["use_onehot_loss"]:
                for k in range(class_num):
                    l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
                    l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])

                    l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
                    l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])

                    l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
                    l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])

                    l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels1[:, k, ...])
                    l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels2[:, k, ...])

                if torch.isnan(l_g_total):
                    import cv2
                    for i in range(images_T1.shape[0]):
                        img1 = images_T1[i].data.cpu().numpy().transpose(1, 2, 0) * 255
                        img2 = images_T2[i].data.cpu().numpy().transpose(1, 2, 0) * 255
                        label1 = labels[i, 0, ...].data.cpu().numpy()
                        label2 = labels[i, 1, ...].data.cpu().numpy()
                        # cv2.imwrite((train_dir + '/img/%d.jpg' % (len(train_img) + i)), trainImg[i])
                        cv2.imwrite(self.config.log_dir + '/error/img1_%d.png' % i, img1.astype('uint8'))
                        cv2.imwrite(self.config.log_dir + '/error/img2_%d.png' % i, img2.astype('uint8'))
                        cv2.imwrite(self.config.log_dir + '/error/label1_%d.png' % i, label1.astype('uint8'))
                        cv2.imwrite(self.config.log_dir + '/error/label2_%d.png' % i, label2.astype('uint8'))
                    raise ValueError('model loss is  none')

            ce_weight=self.ce_weight
            if self.use_abCE:
                l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...], curr_iter=step) * ce_weight + self.cri_ce_loss(pred1[1],
                                                                                                           labels1[:, 1,
                                                                                                           ...],curr_iter=step) * ce_weight
                l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...],curr_iter=step) * ce_weight + self.cri_ce_loss(pred2[1],
                                                                                                           labels2[:, 1,
                                                                                                           ...],curr_iter=step) * ce_weight
                l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...],curr_iter=step) * ce_weight + self.cri_ce_loss(pred3[1],
                                                                                                           labels3[:, 1,
                                                                                                           ...],curr_iter=step) * ce_weight
                l_g_total += self.cri_ce_loss(pred4[0], labels[:, 0, ...],curr_iter=step) * ce_weight + self.cri_ce_loss(pred4[1],
                                                                                                          labels[:, 1,
                                                                                                          ...],curr_iter=step) * ce_weight
            else:
                l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred1[1],
                                                                                                           labels1[:, 1,
                                                                                                           ...]) * ce_weight
                l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2[1],
                                                                                                           labels2[:, 1,
                                                                                                           ...]) * ce_weight
                l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred3[1],
                                                                                                           labels3[:, 1,
                                                                                                           ...]) * ce_weight
                l_g_total += self.cri_ce_loss(pred4[0], labels[:, 0, ...]) * ce_weight*2 + self.cri_ce_loss(pred4[1],
                                                                                                          labels[:, 1,
                                                                                                          ...]) * ce_weight*2
            if self.use_NEloss:
                l_g_NE=0.01*(N8ASCLoss(pred1[0])+N8ASCLoss(pred1[1])+ \
                                  N8ASCLoss(pred2[0]) + N8ASCLoss(pred2[1])+ \
                                  N8ASCLoss(pred3[0]) + N8ASCLoss(pred3[1]))+\
                           0.05 *(N8ASCLoss(pred4[0]) + N8ASCLoss(pred4[1]))
                l_g_total+=l_g_NE

            #===ensure the changed and unchanged part are the same
            if self.train_opt["use_rgb255"]:
                pred1_rgb0 = 1 - F.softmax(pred1[0], 1)[:,0,...].unsqueeze(1)
                pred1_rgb1 = 1 - F.softmax(pred1[1], 1)[:,0,...].unsqueeze(1)
                pred2_rgb0 = 1 - F.softmax(pred2[0], 1)[:,0,...].unsqueeze(1)
                pred2_rgb1 = 1 - F.softmax(pred2[1], 1)[:,0,...].unsqueeze(1)
                pred3_rgb0 = 1 - F.softmax(pred3[0], 1)[:,0,...].unsqueeze(1)
                pred3_rgb1 = 1 - F.softmax(pred3[1], 1)[:,0,...].unsqueeze(1)
                pred4_rgb0 = 1 - F.softmax(pred4[0], 1)[:,0,...].unsqueeze(1)
                pred4_rgb1 = 1 - F.softmax(pred4[1], 1)[:,0,...].unsqueeze(1)
                labels1_f12 = torch.zeros_like(labels1[:, 0, ...])
                labels1_f12[labels1[:, 0, ...] > 0] = 1.0
                labels2_f12 = torch.zeros_like(labels2[:, 0, ...])
                labels2_f12[labels2[:, 0, ...] > 0] = 1.0
                labels3_f12 = torch.zeros_like(labels3[:, 0, ...])
                labels3_f12[labels3[:, 0, ...] > 0] = 1.0
                labels4_f12 = torch.zeros_like(labels[:, 0, ...])
                labels4_f12[labels[:, 0, ...] > 0] = 1.0
                labels1_f12 = labels1_f12.unsqueeze(1).float()
                labels2_f12 = labels2_f12.unsqueeze(1).float()
                labels3_f12 = labels3_f12.unsqueeze(1).float()
                labels4_f12 = labels4_f12.unsqueeze(1).float()
                bce_loss = nn.BCELoss()
                mse_weight = self.config["train"]["ce_weight"]

                l_g_total += bce_loss(pred1_rgb0, labels1_f12) * mse_weight + bce_loss(pred1_rgb1,
                                                                                       labels1_f12) * mse_weight
                l_g_total += bce_loss(pred2_rgb0, labels2_f12) * mse_weight + bce_loss(pred2_rgb1,
                                                                                       labels2_f12) * mse_weight

                l_g_total += bce_loss(pred3_rgb0, labels3_f12) * mse_weight + bce_loss(pred3_rgb1,
                                                                                       labels3_f12) * mse_weight

                l_g_total += bce_loss(pred4_rgb0, labels4_f12) * mse_weight * 2 + bce_loss(pred4_rgb1,
                                                                                           labels4_f12) * mse_weight * 2





        else:
            ce_weight = self.ce_weight
            pred1, pred2 = self.netG(images_T1, images_T2)
            if self.config["train"]["use_onehot_loss"]:
                for k in range(class_num):
                    l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
                    l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
            l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * ce_weight*2 + self.cri_ce_loss(pred2,
                                                                                            labels[:, 1, ...]) * ce_weight*2




        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = 0

    def optimize_parameters_MC7_rgb255(self, step):
        # =============G==================


        self.optimizer_G.zero_grad()
        l_g_total = 0
        images_T12 = self.img
        labels = self.label
        class_num = self.config["network_G_CD"]["out_nc"]
        #one_hot_labels = one_hot_cuda(labels, num_classes=class_num)
        one_hot_labels1=one_hot_cuda(labels[:,0,...],num_classes=class_num)
        one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num)

        images_T1 = images_T12[:, 0:3, ...]
        images_T2 = images_T12[:, 3:6, ...]


        # class_rate1=np.array([0.800605,0.0028,0.0855, 0.0649, 0.0141,0.0314,0.000755])#1==>
        # class_rate2=np.array([0.800605,0.002496, 0.058114, 0.033875, 0.017139, 0.086095, 0.001678])#==>1
        # ====================for Sensetime dataset=================================
        class_weight1 = [ 0.0007,    0.1970,    0.0065,    0.0085,    0.0391,    0.0176,    0.7307]
        class_weight2 = [0.0011,    0.3595,    0.0154,    0.0265,    0.0524,    0.0104,    0.5347]
        #====================for HRSCD=================================
        # class_weight1 = [0.0001,    0.3054,    0.0054,    0.2481, 0,    0.4410]
        # class_weight2 = [0.0001,    0.0097,    0.0458,    0.7787,    0.1557,    0.0100]


        labels_f12 = torch.zeros_like(labels[:, 0, ...])
        labels_f12[labels[:, 0, ...] > 0] = 1.0
        if self.config["train"]["use_DS"]:

            pred1,pred2,pred3,pred4 = self.netG(images_T1, images_T2)
            labels1=F.interpolate(labels.float(),(32,32),mode='bilinear',align_corners=True)
            labels2= F.interpolate(labels.float(), (64, 64), mode='bilinear', align_corners=True)
            labels3= F.interpolate(labels.float(), (128, 128), mode='bilinear', align_corners=True)

            labels1_f12=torch.zeros_like(labels1[:,0,...])
            labels1_f12[labels1[:,0,...]>0]=1.0
            labels2_f12 = torch.zeros_like(labels2[:, 0, ...])
            labels2_f12[labels2[:, 0, ...] > 0] = 1.0
            labels3_f12 = torch.zeros_like(labels3[:, 0, ...])
            labels3_f12[labels3[:, 0, ...] > 0] = 1.0


            labels1,labels2,labels3=labels1.long(),labels2.long(),labels3.long()
            one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num)
            one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num)
            one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num)
            one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num)
            one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num)
            one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num)

            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels2[:, k, ...])

            ce_weight=self.config["train"]["ce_weight"]
            l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred1[1],
                                                                                            labels1[:, 1, ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2[1],
                                                                                            labels2[:, 1, ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred3[1],
                                                                                            labels3[:, 1, ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred4[0], labels[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred4[1],labels[:, 1, ...]) * ce_weight


            seg_weight=self.config["train"]["ce_weight"]
            l_g_total+=self.cri_seg0_loss(pred1[2],labels1_f12.float().unsqueeze(1))*seg_weight+self.cri_seg0_loss(pred2[2],labels2_f12.float().unsqueeze(1))*seg_weight+ \
                       self.cri_seg0_loss(pred3[2], labels3_f12.float().unsqueeze(1)) * seg_weight+self.cri_seg0_loss(pred4[2],labels_f12.float().unsqueeze(1))*seg_weight





        else:
            pred1, pred2,pred12 = self.netG(images_T1, images_T2)
            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
            l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * 10 + self.cri_ce_loss(pred2,
                                                                                            labels[:, 1, ...]) * 10
            l_g_total+=self.cri_seg_loss(pred12,labels_f12.float().unsqueeze(1))*5.0



        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = 0






    def optimize_parameters_SCSeg(self, step):
        # =============G==================
        self.optimizer_G.zero_grad()
        l_g_total = 0
        images_T12 = self.img
        labels = self.label
        class_num = self.config["network_G_CD"]["out_nc"]
        #one_hot_labels = one_hot_cuda(labels, num_classes=class_num)
        label_smooth = self.config["train"]["use_label_smooth"]
        one_hot_labels1=one_hot_cuda(labels[:,0,...],num_classes=class_num,label_smooth=label_smooth)
        one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num,label_smooth=label_smooth)

        images_T1 = images_T12[:, 0:3, ...]
        images_T2 = images_T12[:, 3:6, ...]

        with torch.no_grad():
            _,_,_,images_cd=self.netCD(images_T1,images_T2)

        images_T1=torch.cat([images_T1,images_cd],dim=1)
        images_T2 = torch.cat([images_T2, images_cd], dim=1)

        #=====================for sensetime cd==================
        if self.config.dataset_name=="sensetime":
            class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
            class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        else:
            class_weight1 = [0.0001,    0.3054,    0.0054,    0.2481, 0,    0.4410]
            class_weight2 = [0.0001,    0.0097,    0.0458,    0.7787,    0.1557,    0.0100]
        #======================for franch cd====================
        # class_rate1 = np.array([0.99232, 0.00013, 0.00731, 0.00016, 0, 0.00009])  # 1==>
        # class_rate2 = np.array([0.99232, 0.000803, 0.00017, 0.00001, 0.00005, 0.00078])  # ==>1




        # if self.config["train"]["use_DS"]:
        #
        #     pred1,pred2,pred3,pred4 = self.netG(images_T1, images_T2)
        #     if self.config.patch_size==256:
        #         img_down_size=16
        #     else:
        #         img_down_size = 32
        #     if self.config["train"]["use_progressive_resize"]:
        #         img_down_size=16
        #
        #     labels1=F.interpolate(labels.float(),(img_down_size,img_down_size),mode='bilinear',align_corners=True)
        #     labels2= F.interpolate(labels.float(), (img_down_size*2, img_down_size*2), mode='bilinear', align_corners=True)
        #     labels3= F.interpolate(labels.float(), (img_down_size*4, img_down_size*4), mode='bilinear', align_corners=True)
        #     labels1,labels2,labels3=labels1.long(),labels2.long(),labels3.long()
        #
        #
        #     one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num,label_smooth=label_smooth)
        #     one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num,label_smooth=label_smooth)
        #     one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num,label_smooth=label_smooth)
        #     one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num,label_smooth=label_smooth)
        #     one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num,label_smooth=label_smooth)
        #     one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num,label_smooth=label_smooth)
        #
        #     for k in range(class_num):
        #         l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
        #         l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])
        #
        #         l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
        #         l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])
        #
        #         l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
        #         l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])
        #
        #         l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels1[:, k, ...])
        #         l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels2[:, k, ...])
        #     #print("l_g_total=%.6f" % l_g_total)
        #     if torch.isnan(l_g_total):
        #         import cv2
        #         for i in range(images_T1.shape[0]):
        #             img1=images_T1[i].data.cpu().numpy().transpose(1,2,0)*255
        #             img2=images_T2[i].data.cpu().numpy().transpose(1,2,0)*255
        #             label1=labels[i,0,...].data.cpu().numpy()
        #             label2 = labels[i, 1, ...].data.cpu().numpy()
        #             #cv2.imwrite((train_dir + '/img/%d.jpg' % (len(train_img) + i)), trainImg[i])
        #             cv2.imwrite(self.config.log_dir+'/error/img1_%d.png'% i,img1.astype('uint8'))
        #             cv2.imwrite(self.config.log_dir + '/error/img2_%d.png' % i, img2.astype('uint8'))
        #             cv2.imwrite(self.config.log_dir + '/error/label1_%d.png' % i, label1.astype('uint8'))
        #             cv2.imwrite(self.config.log_dir + '/error/label2_%d.png' % i, label2.astype('uint8'))
        #         raise ValueError('model loss is  none')
        #
        #
        #
        #
        #     ce_weight=self.ce_weight
        #     if self.use_abCE:
        #         l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...], curr_iter=step) * ce_weight + self.cri_ce_loss(pred1[1],
        #                                                                                                    labels1[:, 1,
        #                                                                                                    ...],curr_iter=step) * ce_weight
        #         l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...],curr_iter=step) * ce_weight + self.cri_ce_loss(pred2[1],
        #                                                                                                    labels2[:, 1,
        #                                                                                                    ...],curr_iter=step) * ce_weight
        #         l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...],curr_iter=step) * ce_weight + self.cri_ce_loss(pred3[1],
        #                                                                                                    labels3[:, 1,
        #                                                                                                    ...],curr_iter=step) * ce_weight
        #         l_g_total += self.cri_ce_loss(pred4[0], labels[:, 0, ...],curr_iter=step) * ce_weight + self.cri_ce_loss(pred4[1],
        #                                                                                                   labels[:, 1,
        #                                                                                                   ...],curr_iter=step) * ce_weight
        #     else:
        #         l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred1[1],
        #                                                                                                    labels1[:, 1,
        #                                                                                                    ...]) * ce_weight
        #         l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2[1],
        #                                                                                                    labels2[:, 1,
        #                                                                                                    ...]) * ce_weight
        #         l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred3[1],
        #                                                                                                    labels3[:, 1,
        #                                                                                                    ...]) * ce_weight
        #         l_g_total += self.cri_ce_loss(pred4[0], labels[:, 0, ...]) * ce_weight*2 + self.cri_ce_loss(pred4[1],
        #                                                                                                   labels[:, 1,
        #                                                                                                   ...]) * ce_weight*2
        #     if self.use_NEloss:
        #         l_g_NE=0.01*(N8ASCLoss(pred1[0])+N8ASCLoss(pred1[1])+ \
        #                           N8ASCLoss(pred2[0]) + N8ASCLoss(pred2[1])+ \
        #                           N8ASCLoss(pred3[0]) + N8ASCLoss(pred3[1]))+\
        #                    0.05 *(N8ASCLoss(pred4[0]) + N8ASCLoss(pred4[1]))
        #         l_g_total+=l_g_NE
        #
        #     #===ensure the changed and unchanged part are the same
        #     if self.train_opt["use_rgb255"]:
        #         from models.Satt_CD.modules.utils import LBSign
        #         sign = LBSign.apply
        #         #==logits to label
        #         pred1_rgb0=torch.argmax(F.softmax(pred1[0],1),1)
        #         pred1_rgb1 = torch.argmax(F.softmax(pred1[1], 1),1)
        #         pred2_rgb0 = torch.argmax(F.softmax(pred2[0], 1),1)
        #         pred2_rgb1 = torch.argmax(F.softmax(pred2[1], 1),1)
        #         pred3_rgb0 = torch.argmax(F.softmax(pred3[0], 1),1)
        #         pred3_rgb1 = torch.argmax(F.softmax(pred3[1], 1),1)
        #         pred4_rgb0 = torch.argmax(F.softmax(pred4[0], 1),1)
        #         pred4_rgb1 = torch.argmax(F.softmax(pred4[1], 1),1)
        #
        #         pred1_rgb0 = F.relu(sign(pred1_rgb0 - 0)).float()
        #         pred1_rgb1 = F.relu(sign(pred1_rgb1 - 0)).float()
        #         pred2_rgb0 = F.relu(sign(pred2_rgb0 - 0)).float()
        #         pred2_rgb1 = F.relu(sign(pred2_rgb1 - 0)).float()
        #         pred3_rgb0 = F.relu(sign(pred3_rgb0 - 0)).float()
        #         pred3_rgb1 = F.relu(sign(pred3_rgb1 - 0)).float()
        #         pred4_rgb0 = F.relu(sign(pred4_rgb0 - 0)).float()
        #         pred4_rgb1 = F.relu(sign(pred4_rgb1 - 0)).float()
        #
        #         labels1_f12 = torch.zeros_like(labels1[:, 0, ...])
        #         labels1_f12[labels1[:, 0, ...] > 0] = 1.0
        #         labels2_f12 = torch.zeros_like(labels2[:, 0, ...])
        #         labels2_f12[labels2[:, 0, ...] > 0] = 1.0
        #         labels3_f12 = torch.zeros_like(labels3[:, 0, ...])
        #         labels3_f12[labels3[:, 0, ...] > 0] = 1.0
        #         labels4_f12 = torch.zeros_like(labels[:, 0, ...])
        #         labels4_f12[labels[:, 0, ...] > 0] = 1.0
        #         labels1_f12=labels1_f12.float()
        #         labels2_f12 = labels2_f12.float()
        #         labels3_f12 = labels3_f12.float()
        #         labels4_f12 = labels4_f12.float()
        #
        #         mse_loss=nn.MSELoss()
        #         mse_weight=0.5
        #
        #         l_g_total += mse_loss(pred1_rgb0, labels1_f12) * mse_weight + mse_loss(pred1_rgb1,labels1_f12)* mse_weight
        #         l_g_total += mse_loss(pred2_rgb0, labels2_f12) * mse_weight+ mse_loss(pred2_rgb1, labels2_f12) *  mse_weight
        #
        #         l_g_total += mse_loss(pred3_rgb0, labels3_f12) *  mse_weight + mse_loss(pred3_rgb1,labels3_f12) *  mse_weight
        #
        #         l_g_total += mse_loss(pred4_rgb0, labels4_f12) *  mse_weight*2 + mse_loss(pred4_rgb1,labels4_f12) *  mse_weight*2
        #
        #
        #
        #
        #
        #
        #     #======================for FocalLossWithDice============
        #     # ce_weight=2.0
        #     # d_weight=0.5
        #     # # class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
        #     # # class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        #     # weight=[0.01,3,1,1,1,1,9]
        #     # from models.Satt_CD.modules.loss import FocalLossWithDice
        #     # myLoss=FocalLossWithDice(ce_weight=ce_weight,d_weight=d_weight,weight=weight)
        #     # l_g_total+=myLoss(pred1[0],labels1[:,0,...])+myLoss(pred1[1],labels1[:,1,...])
        #     # l_g_total += myLoss(pred2[0], labels2[:, 0, ...]) + myLoss(pred2[1], labels2[:, 1, ...])
        #     # l_g_total += myLoss(pred3[0], labels3[:, 0, ...]) + myLoss(pred3[1], labels3[:, 1, ...])
        #     # l_g_total += myLoss(pred4[0], labels[:, 0, ...]) + myLoss(pred4[1], labels[:, 1, ...])
        #     #================================================




        #else:

        pred1, pred2 = self.netG(images_T1, images_T2)
        ce_weight=self.config["train"][ "ce_weight"]
        for k in range(class_num):
            l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
            l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
        l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2,
                                                                                        labels[:, 1, ...]) * ce_weight



        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = 0




    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)

            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def get_current_log(self):
        return self.log_dict
    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save_best(self):
        model_path=self.config.model_name
        torch.save(self.netG.state_dict(),model_path)

    def save_best_acc(self):
        #model_path=self.config.model_name
        model_path=self.config.model_dir + '/'+self.config.pred_name+'_best_acc.pth'
        torch.save(self.netG.state_dict(),model_path)

    def save_best_acc_cycle(self,i_cycle):
        #model_path=self.config.model_name
        model_path=self.config.model_dir + '/'+self.config.pred_name+'_best_acc_c'+str(i_cycle)+'.pth'
        torch.save(self.netG.state_dict(),model_path)


    def save_best_loss(self):
        model_path = self.config.model_dir + '/' + self.config.pred_name + '_best_loss.pth'
        torch.save(self.netG.state_dict(),model_path)

    def save(self, iter_step,save_atk=False,save_style=False):
        if save_atk:
            self.save_network(self.netG_atk, 'Gatk', iter_step)
        elif save_style:
            self.save_network(self.netG_AdaIN, 'Gstyle', iter_step)
        else:
            #self.save_network(self.netG, 'G', iter_step)
            model_path = self.config.model_dir + '/' + self.config.pred_name + '_final_iter.pth'
            torch.save(self.netG.state_dict(), model_path)









