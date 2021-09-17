# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by D. F. Peng on 2019/6/18
"""

from utils.utils import mkdir_if_not_exist

import numpy as np
import random
import cv2
import math
from tqdm import tqdm
import glob
import os
import json
import logging
#from osgeo import gdal
#from data_loaders.imgaug16 import  flipud,fliplr,rotate90,rotate180,rotate270,add,mul,gaussian_noise,gaussian_blur,contrast_normal,zoom,translate

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class RSCD_DL(object):
    def __init__(self, config=None):
        #super(RSCD_DL, self).__init__(config) #没有继承类时不要
        #====================load RS data========================

        config.img_dir = config.data_dir + '/result/img'
        config.model_dir = config.data_dir + '/result/model'
        config.log_dir = config.data_dir + '/result/log'
        #==============================for train===================================================

        config.train_dir = config.data_dir + '/train'

        #===============for test=======================


        config.test_dir = config.data_dir + '\\test'
        config.test_pred_dir = config.data_dir + '\\test\\pred'


        mkdir_if_not_exist(config.test_pred_dir)

        if config["model"]=='MRCD':
            # ============for CD=============================
            src_tgt_name = 'Sense-CD'
            print(src_tgt_name)
            # config.pred_name = 'netG2_Res34Res_{}_diffmode_{}_dtype_{}_Drop_{:.2f}_ce_weight_{:.2f}_patch_{}_batch_{}_nepoch_{}_warmepoch_{}'.format(
            # config["network_G_CD"]["which_model_G"],config["network_G_CD"]["diff_mode"],config["network_G_CD"]["dblock_type"],config["train"]["drop_rate"],config["train"]["ce_weight"],
            # config.patch_size, config.batch_size, config["train"]["nepoch"],config["train"]["warmup_epoch"])#for sensetime-cd

            config.pred_name = 'netG1080V3_{}_diffmode_{}_dtype_{}_backbone_{}_patch_{}_batch_{}_nepoch_{}_warmepoch_{}_useDS_{}_useAtt_{}_useOnehotloss_{}'.format(
                config["network_G_CD"]["which_model_G"], config["network_G_CD"]["diff_mode"],
                config["network_G_CD"]["dblock_type"],
                #config["network_G_CD"]["ASPP_type"],
                config["network_G_CD"]["backbone"],
                config.patch_size, config.batch_size, config["train"]["nepoch"], config["train"]["warmup_epoch"],config["network_G_CD"]["use_DS"],
                config["network_G_CD"]["use_att"],config["train"]["use_onehot_loss"])#for whu-cd


        config.pretrained_model_path = config.model_dir + '/' +'netG_Res34Res_EDCls_UNet_BCD_diffmode_diff_dtype_AS_Drop_0.10_ce_weight_4.00_patch_512_batch_4_nepoch_30_warmepoch_0_best_acc.pth'
        if config.mode=="Test":
           config["train"]["mode"]='supervised'



           if config["model"] == 'MRCD':
               if config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU" or config["network_G_CD"]["which_model_G"]=="FC_EF" or config["network_G_CD"]["which_model_G"]=="Seg_EF"or config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU_DCN" or config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU_DCN2":
                   config.pred_dir=config.test_pred_dir +'/'+config.pred_name
                   mkdir_if_not_exist(config.pred_dir)
                   config.precision_path = config.pred_dir + '/acc.txt'
               else:
                   if config.use_CRF:
                       config.pred1_dir = config.test_pred_dir + '\\im1_gray_tta_crf_' + config.pred_name
                       config.pred2_dir = config.test_pred_dir + '\\im2_gray_tta_crf_' + config.pred_name
                       config.pred1_rgb_dir = config.test_pred_dir + '\\im1_rgb_tta_crf_' + config.pred_name
                       config.pred2_rgb_dir = config.test_pred_dir + '\\im2_rgb_tta_crf_' + config.pred_name
                   else:
                       config.pred1_dir = config.test_pred_dir + '\\im1_gray_tta_' + config.pred_name
                       config.pred2_dir = config.test_pred_dir + '\\im2_gray_tta_' + config.pred_name
                       config.pred1_rgb_dir = config.test_pred_dir + '\\im1_rgb_tta_' + config.pred_name
                       config.pred2_rgb_dir = config.test_pred_dir + '\\im2_rgb_tta_' + config.pred_name
                   mkdir_if_not_exist(config.pred1_dir)
                   mkdir_if_not_exist(config.pred2_dir)
                   mkdir_if_not_exist(config.pred1_rgb_dir)
                   mkdir_if_not_exist(config.pred2_rgb_dir)
                   config.precision_path = config.pred1_dir + '/acc.txt'


           else:


               config.pred_dir = config.test_pred_dir + '/pred_' + config.pred_name
               mkdir_if_not_exist(config.pred_dir)
               mkdir_if_not_exist(config.pred_dir+'/Binary')
               config.precision_path = config.pred_dir + '/acc.txt'


        print("pred_model is {}".format(config.pred_name))




        #==============for model======================

        config.model_name = config.model_dir + '/'+config.pred_name+'.pth'

        #==============for log=========================
        config.json_name = config.model_dir + '/min_max.json'
        config.loss_path = config.img_dir + '/' + config.pred_name + '.png'
        config.log_path = config.model_dir + '/' + config.pred_name + '.txt'
        # #============for ramdom mean std===============
        # config.meanA=config.data_dir+'/train/meanA.npy'
        # config.stdA=config.data_dir+'/train/stdA.npy'
        # config.meanB = config.data_dir + '/train/meanB.npy'
        # config.stdB = config.data_dir + '/train/stdB.npy'
        #=============================================
        mkdir_if_not_exist(config.model_dir)
        mkdir_if_not_exist(config.img_dir)
        mkdir_if_not_exist(config.log_dir)
        #===================================================

        self.config=config
        self.data_dir=config.data_dir
        self.train_dir=config.train_dir
        self.test_dir = config.test_dir
        self.val_dir=config.data_dir+'/val'
        self.json_name=config.json_name



    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test




