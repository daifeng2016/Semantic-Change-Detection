import os
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
# import torch modules
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import mkdir_if_not_exist
from data_loaders.RSCD_dl import RSCD_DL
import random
import cv2

#from data_loaders.data_proc import RandomFlip,RandomRotate,RandomShiftScaleRotate,RandomHueSaturationValue,ToTensor,Normalize
#==================================for random augmentation of img an label===================================
#=====================online augmentation: 1) increase augmentation types,  2)save moemory, 3) accelerate training============================
#======================however, the initial sample number should be enough==============================================
mean=[0.4406, 0.4487, 0.4149]
std=[0.1993, 0.1872, 0.1959]
import albumentations as A  # using open-source library for img aug



class RandomCrop(object):
    def __init__(self, ph, pw,scale=1):
        self.ph = ph
        self.pw = pw
        self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h,w=img.shape[:2]#numpy array
        th,tw=int(self.ph*self.scale),int(self.pw*self.scale)
        ix = random.randint(0, w - self.pw)#not using w - self.pw+1
        iy = random.randint(0, h - self.ph)
        tx,ty=int(ix*self.scale),int(iy*self.scale)
        img=img[iy:iy+self.ph, ix:ix+self.pw,:]
        label=label[ty:ty+th, tx:tx+tw]

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]

from models.utils import one_hot_raw
class RandomCropWeight(object):
    def __init__(self, ph, pw):
        self.ph = ph
        self.pw = pw
        #self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h,w=img.shape[:2]#numpy array
        #th,tw=int(self.ph*self.scale),int(self.pw*self.scale)
        bst_x0 = random.randint(0, w - self.pw)#not using w - self.pw+1
        bst_y0 = random.randint(0, h - self.ph)
        one_hot_label=one_hot_raw(label,num_classes=32)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        weight=[1,
                5,5,5,8,5,
                5,5,5,8,5,
                8,8,8,8,8,
                5,5,5,8,5,
                8,8,8,8,8,8,
                5,5,5,5,8
                ]
        for i in range(try_cnt):
            x0 = random.randint(0, w - self.pw)
            y0 = random.randint(0, h - self.ph)
            _sc=0

            for k in range(32):
                _sc+=weight[k]*one_hot_label[y0:y0+self.ph,x0:x0+self.pw,k].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0

        img=img[y0:y0+self.ph, x0:x0+self.pw,:]
        label=label[y0:y0+self.ph,x0:x0+self.pw]

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]


class RandomCropWeight7(object):
    def __init__(self, ph, pw):
        self.ph = ph
        self.pw = pw
        #self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h,w=img.shape[:2]#numpy array
        #th,tw=int(self.ph*self.scale),int(self.pw*self.scale)
        bst_x0 = random.randint(0, w - self.pw)#not using w - self.pw+1
        bst_y0 = random.randint(0, h - self.ph)
        one_hot_label1=one_hot_raw(label[:,:,0],num_classes=7)
        one_hot_label2 = one_hot_raw(label[:,:,1], num_classes=7)
        bst_sc = -1
        #try_cnt = random.randint(1, 14)
        try_cnt = 14
        # weight=[1,
        #         5,5,5,8,5,
        #         5,5,5,8,5,
        #         8,8,8,8,8,
        #         5,5,5,8,5,
        #         8,8,8,8,8,8,
        #         5,5,5,5,8
        #         ]
        # class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
        # class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        # weight1=[1,5,5,5,5,5,10]
        # weight2=[1,5,5,5,5,5,10]

        weight1=[0.01,3.0,1.0,1.0,1.0,1.0,9.0]
        weight2 = [0.01, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]

        for i in range(try_cnt):
            x0 = random.randint(0, w - self.pw)
            y0 = random.randint(0, h - self.ph)
            _sc=0

            for k in range(7):
                _sc+=weight1[k]*one_hot_label1[y0:y0+self.ph,x0:x0+self.pw,k].sum()
                _sc+=weight2[k]*one_hot_label2[y0:y0+self.ph,x0:x0+self.pw,k].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0

        img=img[y0:y0+self.ph, x0:x0+self.pw,:]
        label=label[y0:y0+self.ph,x0:x0+self.pw]

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]


class RandomCropResizeWeight7(object):
    def __init__(self, min_crop_size=100, max_crop_size=480):
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        #self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):

        img, label = sample['img'], sample['label']

        out_crop = random.random() < 0.5
        #out_crop=True
        if out_crop:
            h, w = img.shape[:2]  # numpy array
            crop_size = random.randint(self.min_crop_size, self.max_crop_size)
            #crop_size = random.randint(int(h / 1.15), int(h / 0.85))

            bst_x0 = random.randint(0, w - crop_size)  # not using w - self.pw+1
            bst_y0 = random.randint(0, h - crop_size)
            one_hot_label1 = one_hot_raw(label[:, :, 0], num_classes=7)
            one_hot_label2 = one_hot_raw(label[:, :, 1], num_classes=7)
            bst_sc = -1
            # try_cnt = random.randint(1, 14)
            try_cnt = 14
            # weight=[1,
            #         5,5,5,8,5,
            #         5,5,5,8,5,
            #         8,8,8,8,8,
            #         5,5,5,8,5,
            #         8,8,8,8,8,8,
            #         5,5,5,5,8
            #         ]
            # class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
            # class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
            # weight1=[1,5,5,5,5,5,10]
            # weight2=[1,5,5,5,5,5,10]

            weight1 = [0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]
            weight2 = [0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]

            for i in range(try_cnt):
                x0 = random.randint(0, w - crop_size)
                y0 = random.randint(0, h - crop_size)
                _sc = 0

                for k in range(7):
                    _sc += weight1[k] * one_hot_label1[y0:y0 + crop_size, x0:x0 + crop_size, k].sum()
                    _sc += weight2[k] * one_hot_label2[y0:y0 + crop_size, x0:x0 + crop_size, k].sum()
                if _sc > bst_sc:
                    bst_sc = _sc
                    bst_x0 = x0
                    bst_y0 = y0
            x0 = bst_x0
            y0 = bst_y0

            img_aug = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
            label_aug = label[y0:y0 + crop_size, x0:x0 + crop_size,:]

            img_aug = cv2.resize(img_aug, (h, w), interpolation=cv2.INTER_CUBIC)
            label_aug = cv2.resize(label_aug, (h, w), interpolation=cv2.INTER_LINEAR)# cannot set cv2.INTER_CUBIC
            return {'img': img_aug, 'label': label_aug, 'name': sample['name']}

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]

class RandomCutMix(object):
    def __init__(self, min_crop_size=64, max_crop_size=480,num_class=7):
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        #self.scale=scale#denote to the upsampling scale
        self.num_class=num_class
    def __call__(self, sample):

        img, label = sample['img'], sample['label']

        out_crop = random.random() < 0.5
        #out_crop=True
        if out_crop:
            h, w = img.shape[:2]  # numpy array
            crop_size = random.randint(self.min_crop_size, self.max_crop_size)
            #crop_size = random.randint(int(h / 1.15), int(h / 0.85))

            bst_x0 = random.randint(0, w - crop_size)  # not using w - self.pw+1
            bst_y0 = random.randint(0, h - crop_size)
            one_hot_label1 = one_hot_raw(label[:, :, 0], num_classes=self.num_class)
            one_hot_label2 = one_hot_raw(label[:, :, 1], num_classes=self.num_class)
            bst_sc = -1
            # try_cnt = random.randint(1, 14)
            try_cnt = 14


            weight1 = [0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]
            weight2 = [0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]

            for i in range(try_cnt):
                x0 = random.randint(0, w - crop_size)
                y0 = random.randint(0, h - crop_size)
                _sc = 0

                for k in range(self.num_class):
                    _sc += weight1[k] * one_hot_label1[y0:y0 + crop_size, x0:x0 + crop_size, k].sum()
                    _sc += weight2[k] * one_hot_label2[y0:y0 + crop_size, x0:x0 + crop_size, k].sum()
                if _sc > bst_sc:
                    bst_sc = _sc
                    bst_x0 = x0
                    bst_y0 = y0
            x0 = bst_x0
            y0 = bst_y0

            #=====use cutmix for aug==============
            img1,img2=img[:,:,:3],img[:,:,3:]
            label1,label2=label[:,:,:1],label[:,:,1:]
            img1_aug,img2_aug=img1.copy(),img2.copy()
            label1_aug,label2_aug=label1.copy(),label2.copy()
            img1_aug[y0:y0 + crop_size, x0:x0 + crop_size, :]=img2[y0:y0 + crop_size, x0:x0 + crop_size, :]
            img2_aug[y0:y0 + crop_size, x0:x0 + crop_size, :] = img1[y0:y0 + crop_size, x0:x0 + crop_size, :]
            label1_aug[y0:y0 + crop_size, x0:x0 + crop_size, :] = label2[y0:y0 + crop_size, x0:x0 + crop_size, :]
            label2_aug[y0:y0 + crop_size, x0:x0 + crop_size, :] = label1[y0:y0 + crop_size, x0:x0 + crop_size, :]

            img_aug = np.concatenate([img1_aug,img2_aug],2)
            label_aug = np.concatenate([label1_aug,label2_aug],2)


            return {'img': img_aug, 'label': label_aug, 'name': sample['name']}

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]


class RandomMixorScale(object):
    def __init__(self, num_class=7):
        self.num_class=num_class
    def __call__(self, sample):
        img,label = sample['img'],sample['label']
        use_mix=random.random() < 0.5
        #use_mix = random.random() < 0.3#use scale more
        if use_mix:
            aug=RandomCutMix()
            aug_res = aug(sample)
            img,label=aug_res['img'],aug_res['label']
        else:
            aug=RandomScale(use_CD=True)
            aug_res = aug(sample)
            img, label = aug_res['img'], aug_res['label']





        return {'img':img,'label':label,'name':sample['name']}




class ReSize(object):
    def __init__(self, ph, pw,scale=1):
        self.ph = ph
        self.pw = pw
        self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img_new=cv2.resize(img,(self.ph,self.pw),interpolation=cv2.INTER_CUBIC)
        label_new=cv2.resize(label,(self.ph,self.pw),interpolation=cv2.INTER_CUBIC)

        return {'img':img_new,'label':label_new,'name':sample['name']}


class RandomFlip(object):
    def __call__(self, sample):
        img,label = sample['img'],sample['label']
        hflip=random.random() < 0.5
        vflip=random.random() < 0.5
        #dfilp=random.random() < 0.5

        if vflip:
            img= np.flipud(img).copy()
            label=np.flipud(label).copy()
        if hflip:
            img= np.fliplr(img).copy()
            label = np.fliplr(label).copy()
        # if dfilp:
        #     img=cv2.flip(img,-1)
        #     label = cv2.flip(label, -1)

        return {'img':img,'label':label,'name':sample['name']}
class RandomRotate(object):
    def __call__(self, sample):
        img, label = sample['img'],sample['label']
        rot90 = random.random() < 0.5
        rot180 = random.random() < 0.5
        rot270 = random.random() < 0.5

        if rot90:
            img = np.rot90(img,1).copy()
            label=np.rot90(label,1).copy()
        if rot180:
            img = np.rot90(img,2).copy()
            label = np.rot90(label,2).copy()
        if rot270:
            img=np.rot90(img,3).copy()
            label = np.rot90(label,3).copy()

        return {'img':img,'label':label,'name':sample['name']}
class RandomHueSaturationValue(object):
    def __call__(self, sample, hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15), u=0.5):
        img1, label = sample['img'], sample['label']
        if np.random.random() < u:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            h1, s1, v1 = cv2.split(img1)

            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            # h2, s2, v2 = cv2.split(img2)

            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)

            h1 += hue_shift
            #h2 += hue_shift

            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s1 = cv2.add(s1, sat_shift)
            #s2 = cv2.add(s2, sat_shift)

            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v1 = cv2.add(v1, val_shift)
            #v2 = cv2.add(v2, val_shift)

            img1 = cv2.merge((h1, s1, v1))
            img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)

            # img2 = cv2.merge((h2, s2, v2))
            # img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)


        return {'img':img1,'label':label,'name':sample['name']}

class RandomShiftScaleRotate(object):

        def __call__(self, sample, shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
          img1, label = sample['img'], sample['label']
          if np.random.random() < u:
            height, width, channel = img1.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)


            img1= cv2.warpPerspective(img1, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            # img2 = cv2.warpPerspective(img2, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
            #                            borderValue=(
            #                                0, 0,
            #                                0,))
            label = cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))

          return {'img': img1, 'label': label, 'name': sample['name']}


#============================================================================================================
#=================================for random augmentation of T12=============================================
#============================================================================================================
class RandomFlipT12(object):
    def __call__(self, sample):
        img_T1, img_T2,label = sample['imgT1'], sample['imgT2'],sample['label']
        hflip=random.random() < 0.5
        vflip=random.random() < 0.5
        dfilp=random.random() < 0.5
        # if random.random() > 0.5:
        #     img_LR = np.fliplr(img_LR).copy()
        #     img_HR = np.fliplr(img_HR).copy()
        if vflip:
            img_T1 = np.flipud(img_T1).copy()
            img_T2 = np.flipud(img_T2).copy()
            label=np.flipud(label).copy()
        if hflip:
            img_T1 = np.fliplr(img_T1).copy()
            img_T2 = np.fliplr(img_T2).copy()
            label = np.fliplr(label).copy()
        if dfilp:
            img_T1=cv2.flip(img_T1,-1)
            img_T2 = cv2.flip(img_T2, -1)
            label = cv2.flip(label, -1)

        return {'imgT1':img_T1,'imgT2':img_T2,'label':label}

class RandomRotateT12(object):
    def __call__(self, sample):
        img_T1, img_T2,label = sample['imgT1'], sample['imgT2'],sample['label']
        rot90 = random.random() < 0.5
        rot180 = random.random() < 0.5
        rot270 = random.random() < 0.5

        if rot90:
            img_T1 = np.rot90(img_T1,1).copy()
            img_T2 = np.rot90(img_T2,1).copy()
            label=np.rot90(label,1).copy()
        if rot180:
            img_T1 = np.rot90(img_T1,2).copy()
            img_T2 = np.rot90(img_T2,2).copy()
            label = np.rot90(label,2).copy()
        if rot270:
            img_T1=np.rot90(img_T1,3).copy()
            img_T2 = np.rot90(img_T2,3).copy()
            label = np.rot90(label,3).copy()

        return {'imgT1':img_T1,'imgT2':img_T2,'label':label}

class RandomShiftScaleRotateT12(object):
    def __call__(self, sample,shift_limit=(-0.0, 0.0),
                               scale_limit=(-0.0, 0.0),
                               rotate_limit=(-0.0, 0.0),
                               aspect_limit=(-0.0, 0.0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
        img_T1, img_T2,label = sample['imgT1'], sample['imgT2'],sample['label']
        if random.random()<u:
            height, width, channel = img_T1.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)

            img1 = cv2.warpPerspective(img_T1, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))
            img2 = cv2.warpPerspective(img_T2, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))
            mask = cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))

        return {'imgT1':img_T1,'imgT2':img_T2,'label':label}

class RandomHueSaturationValueT12(object):
    def __call__(self, sample,hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
        img_T1, img_T2, label = sample['imgT1'], sample['imgT2'],sample['label']
        if random.random()<u:
            img1 = cv2.cvtColor(img_T1, cv2.COLOR_BGR2HSV)
            h1, s1, v1 = cv2.split(img1)

            img2 = cv2.cvtColor(img_T2, cv2.COLOR_BGR2HSV)
            h2, s2, v2 = cv2.split(img2)

            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)

            h1 += hue_shift
            h2 += hue_shift

            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s1 = cv2.add(s1, sat_shift)
            s2 = cv2.add(s2, sat_shift)

            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v1 = cv2.add(v1, val_shift)
            v2 = cv2.add(v2, val_shift)

            img1 = cv2.merge((h1, s1, v1))
            img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)

            img2 = cv2.merge((h2, s2, v2))
            img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)

            img_T1=img1.copy()
            img_T2=img2.copy()

        return  {'imgT1':img_T1,'imgT2':img_T2,'label':label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __call__(self, sample):
        img_T1, img_T2, label = sample['imgT1'], sample['imgT2'], sample['label']
        label=np.expand_dims(label,axis=-1)
        img_T1_tensor=torch.from_numpy(img_T1.transpose((2,0,1)))
        img_T2_tensor= torch.from_numpy(img_T2.transpose((2, 0, 1)))
        label_tensor = torch.from_numpy(label.transpose((2, 0, 1)))
        img_T1_tensor= img_T1_tensor.float().div(255)
        img_T2_tensor= img_T2_tensor.float().div(255)
        label_tensor=label_tensor.float().div(255)
        return {'imgT1':img_T1_tensor,'imgT2':img_T2_tensor,'label':label_tensor}
class ToTensor_Test(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __call__(self, sample):
        img_T12= sample['image']

        img_T12_tensor=torch.from_numpy(img_T12.transpose((2,0,1)))

        img_T12_tensor= img_T12_tensor.float().div(255)
        return {'image':img_T12_tensor,'name':sample['name']}



class ToTensor_Sense(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __init__(self,use_label255=False,use_rgb=False,use_label32=False):


        # self.use_label_rgb=use_label_rgb
        self.use_label32=use_label32
        self.use_label255=use_label255
        self.use_rgb=use_rgb
    def __call__(self, sample):
        img= sample['img']
        timg=torch.from_numpy(img.transpose((2,0,1)))#[512,512,6]==>[6,512,512]
        timg= timg.float().div(255)

        label = sample['label']#[512,512,2]
        if isinstance(label,np.ndarray):
            if self.use_label255:
                tlabel = torch.from_numpy(label.transpose((2, 0, 1)))
                tlabel = tlabel.float().div(255)
            elif self.use_rgb:
                tlabel = torch.from_numpy(label.transpose((2, 0, 1)))
            else:
                tlabel = torch.from_numpy(label)
                #tlabel = torch.from_numpy(label.transpose((2, 0, 1)))
        else:
            #tlabel=None# cause exception using none
            tlabel = torch.from_numpy(label)
        return {'img':timg,'label':tlabel,'name':sample['name']}



class Normalize_BR(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __init__(self, mean, std):
        # self.mean = torch.from_numpy(np.array(mean))
        # self.std =torch.from_numpy(np.array(std))
        self.mean=mean
        self.std=std

    def __call__(self, sample):
        img= sample['img']
        # img-=self.mean
        # img/=self.std
        self.mean = torch.Tensor(self.mean).view(3, 1, 1)#must have .view(3, 1, 1), broadcat can work only the two tensors have same ndims
        self.std=torch.Tensor(self.std).view(3, 1, 1)
        img=img.sub_(self.mean).div_(self.std)


        return {'img':img,'label':sample['label'],'name':sample['name']}





class Normalize(object):
    def __call__(self, sample,mean,std):
        img_T1, img_T2, label = sample['imgT1'], sample['imgT2'], sample['label']
        img_T1-=mean[0:3]
        img_T1/=std[0:3]
        img_T2-=mean[3:6]
        img_T2/=std[3:6]
        return {'imgT1': img_T1, 'imgT2': img_T2, 'label': label}
#####################################################################
#========================for data loader============================#
#####################################################################
transform = transforms.Compose([
    # RandomShiftScaleRotate(),
    # RandomFlip(),
    # RandomRotate(),
    # ToTensor()#note that new dict is used for input, hence transforms.ToTensor()  should not be used

    transforms.ToTensor()
    #transforms.Normalize(mean=[.5,.5,.5,.5,.5,.5],std=[.5,.5,.5,.5,.5,.5])#[-1,1]
])


target_transform = transforms.Compose([
    transforms.ToTensor()  ##将图片转换为Tensor,归一化至[0,1]
])




#===============for dataloader speeding=================
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            #self.next_input = self.next_input.sub_(self.mean).div_(self.std)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

class ImagesetDatasetCD(Dataset):
    """ Derived Dataset class for loading many imagesets from a list of directories."""

    def __init__(self, imset_list, config, mode='Train',seed=None,transform=None):

        super().__init__()
        self.dl = RSCD_DL(config=config)
        self.imset_list = imset_list
        #self.imset_dir=config.data_dir+'/train/aug7'
        self.imset_dir = config.train_dir
        self.test_dir=config.test_dir

        self.seed = seed  # seed for random patches
        self.mode=mode
        self.transform=transform
        self.out_class=config["network_G_CD"]["out_nc"]
        self.config=config

    def __len__(self):
        if self.mode=="Train":
            repeat=1
            if self.config.iter_per_epoch>800:
               repeat = self.config.batch_size * self.config.iter_per_epoch // len(self.imset_list)
        else:
            repeat = 1
        return len(self.imset_list)*repeat
        #return len(self.imset_list)

    def __getitem__(self, index):

        if self.mode == 'Train' or self.mode=='Val':
            index = (index % len(self.imset_list))
            cur_dir = self.imset_dir
        else:
            cur_dir = self.test_dir


        img1_path = cur_dir + '/T1/' + self.imset_list[index]
        index0 = self.imset_list[index].rfind('.', 0, len(self.imset_list[index]))
        img_name = self.imset_list[index][0:index0]
        img2_path = cur_dir + '/T2/' + self.imset_list[index]

        label_path = cur_dir + '/label/' + self.imset_list[index]

        # process the images
        img1 = np.asarray(Image.open(img1_path).convert('RGB'))
        img2=np.asarray(Image.open(img2_path).convert('RGB'))
        img12 = np.concatenate([img1, img2], 2)

        label = np.asarray(Image.open(label_path).convert('L'))

        sample = {'img': img12, 'label': label, 'name': img_name}
        if self.transform is not None:
            sample_train = self.transform(sample)  # must convert to torch tensor
        if self.out_class >1:
            sample_train['label']=sample_train['label'].long().squeeze(0)
            #return inputImage, targetImage.long().squeeze(0)  # for 2-channel output

        return sample_train



class ImagesetDatasetCD_Sense(Dataset):
    """ Derived Dataset class for loading many imagesets from a list of directories."""

    def __init__(self, imset_list, config, mode='Train',use_score=False,use_label_rgb=False,use_label32=False,use_label255=False,label_type=None,seed=None,transform=None):

        super().__init__()
        self.dl = RSCD_DL(config=config)
        self.imset_list = imset_list
        #self.imset_dir=config.data_dir+'/train/aug7'
        self.imset_dir = config.train_dir
        self.test_dir=config.test_dir

        self.seed = seed  # seed for random patches
        self.mode=mode
        self.transform=transform
        self.out_class=config["network_G_CD"]["out_nc"]
        self.config=config
        self.use_label_rgb=use_label_rgb
        self.use_label32=use_label32
        self.use_label255=use_label255
        if label_type=='label32':
            self.use_label32=True
        elif label_type=='label255':
            self.use_label255=True
        else:
            self.use_label_rgb=True


        self.use_score=use_score

    def __len__(self):
        if self.mode=="Train":
            repeat=1
            # if self.config.iter_per_epoch>800:# if not using aug6
            #    repeat = self.config.batch_size * self.config.iter_per_epoch // len(self.imset_list)
            #    if repeat==0:
            #        repeat=1
            if self.config.iter_per_epoch>(len(self.imset_list)//self.config.batch_size):
               repeat = self.config.batch_size * self.config.iter_per_epoch /len(self.imset_list)
        else:
            repeat = 1
        return int(len(self.imset_list)*repeat)
        #return len(self.imset_list)

    def __getitem__(self, index):

        if self.mode == 'Train' or self.mode=='Val':
            index = (index % len(self.imset_list))
            cur_dir = self.imset_dir
        else:
            cur_dir = self.test_dir

        if self.config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD":
            img1_path = cur_dir + '/im1/' + self.imset_list[index]
            img2_path = cur_dir + '/im2/' + self.imset_list[index]
        else:
            img1_path = cur_dir + '/im1/' + self.imset_list[index]
            img2_path = cur_dir + '/im2/' + self.imset_list[index]
        #img1_path = cur_dir + '/im1/' + self.imset_list[index]
        index0 = self.imset_list[index].rfind('.', 0, len(self.imset_list[index]))
        img_name = self.imset_list[index][0:index0]
        #img2_path = cur_dir + '/im2/' + self.imset_list[index]
        # process the images
        img1 = np.asarray(Image.open(img1_path).convert('RGB'))
        img2=np.asarray(Image.open(img2_path).convert('RGB'))
        img12 = np.concatenate([img1, img2], 2)

        if self.mode == 'Train' or self.mode == 'Val' or self.use_score:
            if self.use_label_rgb:
                label1_path = cur_dir + '/label1/' + self.imset_list[index]
                label2_path = cur_dir + '/label2/' + self.imset_list[index]
                label1 = np.asarray(Image.open(label1_path).convert('L'))[:, :, None]
                label2 = np.asarray(Image.open(label2_path).convert('L'))[:, :, None]

                label = np.concatenate([label1, label2], 2)
                sample = {'img': img12, 'label': label, 'name': img_name}
            if self.use_label32:
                label_path = cur_dir + '/label32/' + self.imset_list[index]
                label = np.asarray(Image.open(label_path).convert('L'))
                #label = np.asarray(Image.open(label_path).convert('L'))[:, :, None]
            if self.use_label255:
                label_path = cur_dir + '/label255/' + self.imset_list[index]
                label = np.asarray(Image.open(label_path).convert('L'))[:, :, None]#(512,512)==>(512,512,1)

        else:

            label=np.zeros((img1.shape[0],img1.shape[1],1),dtype='uint8')

        sample = {'img': img12, 'label': label, 'name': img_name}
        if self.transform is not None:
            sample_train = self.transform(sample)  # must convert to torch tensor

        if self.out_class >1:
            if isinstance(sample_train['label'],torch.Tensor):
               sample_train['label'] = sample_train['label'].long()



        return sample_train








class TrainDatasetCycle(Dataset):
    def __init__(self, config,
                 #transform=transform, target_transform=targetTransform,
                 mode='Train'):
        #self.rootDir = rootDir
        self.transform = transform
        self.targetTransform = target_transform
        #self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ')
        self.dl=RSCD_DL(config=config)
        self.mode=mode
        self.train_set, self.val_set = self.dl.get_train_val(val_rate=0.1)


        train_numb = len(self.train_set)
        valid_numb = len(self.val_set)

        print("the number of train data is", train_numb)
        print("the number of val data is", valid_numb)

        self.test_set=self.dl.get_test()
        test_numb = len(self.test_set)
        print("the number of test data is", test_numb)
        ######################################################################################################################
        #===========Now here we equalize the lengths of labelled and unlabelled imgs by just repeating up some images=========
        self.ratio=train_numb*1.0/(train_numb+test_numb)
        if self.ratio > 0.5:
            new_ratio = round((self.ratio / (1 - self.ratio + 1e-6)), 1)
            excess_ratio = new_ratio - 1
            new_list_1 = self.test_set * int(excess_ratio)
            new_list_2 = list(np.random.choice(np.array(self.test_set),
                                               size=int((excess_ratio - int(excess_ratio)) *self.test_set.__len__()),
                                               replace=False))
            self.test_set += (new_list_1 + new_list_2)
        elif self.ratio < 0.5:
            new_ratio = round(((1 - self.ratio) / (self.ratio + 1e-6)), 1)
            excess_ratio = new_ratio - 1
            new_list_1 = self.train_set * int(excess_ratio)
            new_list_2 = list(np.random.choice(np.array(self.train_set),
                                               size=int((excess_ratio - int(excess_ratio)) * self.train_set.__len__()),
                                               replace=False))
            self.train_set += (new_list_1 + new_list_2)


    def __len__(self):
        if self.mode=='Train':
            return len(self.train_set)
        elif self.mode=='Val':
            return len(self.val_set)  # 需要重写 len 方法，该方法提供了dataset的大小
        else:
            return len(self.test_set)

    def __getitem__(self, idx):
        # input and target images
        if self.mode=='Train':
           img_T1_path = self.dl.train_dir + '/T1/' + self.train_set[idx]
           (filepath, filename) = os.path.split(img_T1_path)
           img_T2_path = self.dl.train_dir + '/T2/' + filename
           label_path = self.dl.train_dir + '/label/' + filename
        elif self.mode=='Val':
            img_T1_path = self.dl.train_dir + '/T1/' + self.val_set[idx]
            (filepath, filename) = os.path.split(img_T1_path)
            img_T2_path = self.dl.train_dir + '/T2/' + filename
            label_path = self.dl.train_dir + '/label/' + filename
        else:
            img_T1_path = self.dl.train1_dir + '/T1/' + self.test_set[idx]
            (filepath, filename) = os.path.split(img_T1_path)
            img_T2_path = self.dl.train1_dir + '/T2/' + filename
            #label_path = self.dl.train1_dir + '/label/' + filename

        # process the images
        img_T1 = np.asarray(Image.open(img_T1_path).convert('RGB'))
        img_T2 = np.asarray(Image.open(img_T2_path).convert('RGB'))
        #inputImage=np.concatenate([np.transpose(img_T1,(2,0,1)),np.transpose(img_T2,(2,0,1))],0)
        inputImage = np.concatenate([img_T1, img_T2], 2)
        if self.transform is not None:
             inputImage= self.transform(inputImage)  # 输入的原始通道图像，自动转化为torch tensor [nYsize,nXsize,channel]==>[channel,nYsize,nXsize]

        if self.mode=='Train' or self.mode=='Val':

           targetImage = Image.open(label_path).convert('L')
           if self.targetTransform is not None:
             targetImage = self.targetTransform(targetImage)
           # myOneHot=OneHotEncode()#是类，使用前必须实例化
           # ohlabel=myOneHot(targetImage)
           return inputImage, targetImage.long()#for two-channel output
           #return inputImage, targetImage
        else:
            return inputImage


#==============================================================================
class OneHotEncode(object):
    """
        Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    """
    def __init__(self,nclass=2):
        self.nclass = nclass

    def __call__(self,label):
        label_a = np.array((label.byte().squeeze(0)),np.uint8)

        ohlabel = np.zeros((self.nclass,label_a.shape[0],label_a.shape[1])).astype(np.uint8)

        for c in range(self.nclass):
            ohlabel[c:,:,:] = (label_a == c).astype(np.uint8)

        return torch.from_numpy(ohlabel)
#=====================random augmentation for img12 using import albumentations as A=============================
import albumentations as A
class RandomShiftScale(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']


        if self.use_CD:
            # aug = A.ShiftScaleRotate(p=1)#p must be 1 , or else aug_op for img1 and img2 may be different
            # img1_aug = aug(image=img[:, :, :3], mask=label)
            # img2_aug = aug(image=img[:, :, 3:], mask=label)
            # img_aug = np.concatenate([img1_aug['image'], img2_aug['image']], 2)
            # label_aug = img1_aug['mask']
            aug=A.Compose([
                 A.ShiftScaleRotate(0.5)],additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3], image2=img[:, :, 3:], mask=label)
            # img1_aug = aug_op['image']
            # img2_aug = aug_op['image2']
            img_aug = np.concatenate([aug_op['image'], aug_op['image2']], 2)
            label_aug = aug_op['mask']

        else:
            aug = A.ShiftScaleRotate(p=0.5)
            aug_f = aug(image=img, mask=label)
            img_aug=aug_f['image']
            label_aug=aug_f['mask']

        return {'img': img_aug, 'label': label_aug, 'name': sample['name']}

class RandomScale(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']


        if self.use_CD:

            # aug=A.Compose([
            #      A.RandomScale(p=0.5)],additional_targets={'image2': 'image'})#will change input size

            aug = A.Compose([
                A.RandomSizedCrop(min_max_height=(100, 500), height=512, width=512, interpolation=2, p=0.5)], additional_targets={'image2': 'image'})

            aug_op = aug(image=img[:, :, :3], image2=img[:, :, 3:], mask=label)

            img_aug = np.concatenate([aug_op['image'], aug_op['image2']], 2)
            label_aug = aug_op['mask']

        else:
            aug = A.RandomSizedCrop(min_max_height=(50, 500), height=512, width=512, interpolation=2, p=0.5)
            aug_f = aug(image=img, mask=label)
            img_aug=aug_f['image']
            label_aug=aug_f['mask']

        return {'img': img_aug, 'label': label_aug, 'name': sample['name']}





class RandomTranspose(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']


        if self.use_CD:

            aug = A.Compose([
                A.Transpose(p=0.5)], additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3], image2=img[:, :, 3:], mask=label)
            img_aug = np.concatenate([aug_op['image'], aug_op['image2']], 2)
            label_aug = aug_op['mask']
        else:
            aug = A.Transpose(p=0.5)
            aug_f = aug(image=img, mask=label)
            img_aug = aug_f['image']
            label_aug = aug_f['mask']


        return {'img': img_aug, 'label': label_aug, 'name': sample['name']}


class RandomNoise(object):#can not use oneof for imgs12, which may lead to differnt aug_op for img1 and img2
    def __init__(self, use_CD=False):
        self.use_CD = use_CD
    def __call__(self, sample):
        img, label = sample['img'], sample['label']



        if self.use_CD:
            aug = A.Compose([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise()], additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3])
            img_aug = np.concatenate([aug_op['image'],img[:,:,3:]], 2)
            #label1_aug = label
        else:
            aug = A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.5)
            aug_f = aug(image=img)
            img_aug=aug_f['image']


        return {'img': img_aug, 'label': label, 'name': sample['name']}

class RandomColor(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD
    def __call__(self, sample):
        img, label = sample['img'], sample['label']



        if self.use_CD:
            aug = A.Compose([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss()], additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3])
            img_aug = np.concatenate([aug_op['image'], img[:, :, 3:]], 2)
            #label1_aug = label
        else:
            aug = A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss()
            ], p=0.5)
            aug_f=aug(image=img)
            img_aug=aug_f['image']


        return {'img': img_aug, 'label': label, 'name': sample['name']}


class RandomColor2(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']



        if self.use_CD:
            aug = A.Compose([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
                A.RGBShift()], additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3])
            img_aug = np.concatenate([aug_op['image'], img[:, :, 3:]], 2)
            #label1_aug = label
        else:
            aug = A.OneOf([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
                A.RGBShift()
            ], p=0.5)
            aug_f = aug(image=img)
            img_aug = aug_f['image']

        return {'img': img_aug, 'label': label, 'name': sample['name']}


class RandomMix(object):
    # def __init__(self, use_CD=False):
    #     self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        b_mix = random.random() < 0.5
        f_mix = random.random() < 0.5
        # mix two periods of images using label map
        img1 = img[:, :, :3]
        label1 = label[:, :, 0]#[512,512],label[:, :, :1]==>[512,512,1]
        img2 = img[:, :, 3:]
        label2 = label[:, :, 1]
        if b_mix:
            # change background, label_mix is unchanged
            img1_mix = img1.copy()
            label1_mix = label1.copy()
            img2_mix = img2.copy()
            label2_mix = label2.copy()

            img1_mix[label1 == 0] = img2[label2 == 0]
            img2_mix[label2 == 0] = img1[label2 == 0]

            img= np.concatenate([img1_mix, img2_mix], 2)
            label= np.concatenate([label1_mix[:, :, None], label2_mix[:, :, None]], 2)
            return {'img': img, 'label': label, 'name': sample['name']}
        if f_mix:
            # change foreground, label_mix is changed
            img1_mix = img1.copy()
            label1_mix = label1.copy()
            img2_mix = img2.copy()
            label2_mix = label2.copy()

            img1_mix[label1 > 0] = img2[label2 > 0]
            img2_mix[label2 > 0] = img1[label2 > 0]
            label1_mix[label1 > 0] = label2[label1 > 0]
            label2_mix[label2 > 0] = label1[label2 > 0]

            img= np.concatenate([img1_mix, img2_mix], 2)
            label= np.concatenate([label1_mix[:, :, None], label2_mix[:, :, None]], 2)
            return {'img': img, 'label': label, 'name': sample['name']}


        return {'img': img, 'label': label, 'name': sample['name']}


