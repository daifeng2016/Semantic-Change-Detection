import torch
import  numpy as np
import cv2
from utils.utils import mkdir_if_not_exist
def calculate_mean_std(data_loader,mode="BI"):



   mean = 0
   std = 0
   nb_samples = 0
   if mode == 'BI':
       # ==========compute mean std based on each batch=================================
       for batch_idx, dataset_dict in enumerate(data_loader, 0):  # len(data_loader)
           # print("\rwriting training aug %d patch" % (img_count + i), end='')
           print("\rcalculate batch %d" % (batch_idx + 1), end='')
           batch_samples = dataset_dict['img'].size(0)  # check this line for correctness
           imgs = dataset_dict['img'].double().view(batch_samples, dataset_dict['img'].size(1), -1)
           # print(imgs.size())
           mean += imgs.mean(2).sum(0)
           std += imgs.std(2).sum(0)
           nb_samples += batch_samples
       # print(nb_samples)
       mean /= nb_samples
       std /= nb_samples
   else:
       mean = 0.0
       for batch_idx, dataset_dict in enumerate(data_loader, 0):
           print("\rcalculate batch %d" % (batch_idx + 1), end='')
           images = dataset_dict['img']
           batch_samples = images.size(0)
           images = images.view(batch_samples, images.size(1), -1)
           mean += images.mean(2).sum(0)
           nb_samples += batch_samples
       mean = mean / nb_samples

       var = 0.0
       print("\nvar...")
       for batch_idx, dataset_dict in enumerate(data_loader, 0):
           print("\rcalculate batch %d" % (batch_idx + 1), end='')
           images = dataset_dict['img']
           batch_samples = images.size(0)
           images = images.view(batch_samples, images.size(1), -1)
           var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])  # [4,3,16384]-[3,1]=[4,3,16384].sum([0,2])=[3]

       std = torch.sqrt(var / (nb_samples * images.size(2)))  # torch.sqrt(var / (nb_samples))

   print("mean=", mean)
   print("std=", std)

   return mean, std

def get_idx(change_type,cur_change):
    idx=-1
    for i in range(len(change_type)):
        if cur_change == change_type[i]:
            idx = i
            break
    if idx==-1:
        print("%s not found" % cur_change)
    return idx


def compute_comb_label(config,data_loader):

    # change_type=['0_0',
    #              '1_1','1_2','1_3','1_4','1_5','1_6',
    #              '2_1','2_2', '2_3', '2_4', '2_5', '2_6',
    #              '3_1', '3_2', '3_3','3_4', '3_5', '3_6',
    #              '4_1', '4_2', '4_3','4_4', '4_5', '4_6',
    #              '5_1', '5_2', '5_3', '5_4', '5_5','5_6',
    #              '6_1', '6_2', '6_3', '6_4', '6_5','6_6']

    # change_type = ['0_0',
    #                 '1_2', '1_3', '1_4', '1_5', '1_6',
    #                '2_1', '2_3', '2_4', '2_5', '2_6',
    #                '3_1', '3_2',  '3_4', '3_5', '3_6',
    #                '4_1', '4_2', '4_3',  '4_5', '4_6',
    #                '5_1', '5_2', '5_3', '5_4', '5_5', '5_6',
    #                '6_1', '6_2', '6_3', '6_4', '6_5']#for sensetime dataset
    change_type = ['0_0',
                   '1_2',  '1_4', '1_5',
                   '2_1', '2_3', '2_5',
                   '3_1', '3_2',
                   '5_1', '5_2', '5_4']#for HRSCD dataset

    label_sta={}
    for i in range(len(change_type)):
        label_sta[change_type[i]]=[]
    img_w,img_h=512,512
    label_binary_dir=config.train_dir+'/label255'
    label_com_dir=config.train_dir+'/label32'
    mkdir_if_not_exist(label_binary_dir)
    mkdir_if_not_exist(label_com_dir)

    for batch_idx, dataset_dict in enumerate(data_loader, 0):
        #print("\rcalculate batch %d" % (batch_idx + 1), end='')
        label12 = dataset_dict['label']
        img_name = dataset_dict['name']
        print("calculate %d image %s" % (batch_idx + 1,img_name[0]))

        label1=label12[0,0,...].numpy()
        label2 = label12[0, 1, ...].numpy()
        label_new=np.zeros_like(label1)

        # label_binary=np.zeros_like(label1)
        # label_binary[label1>0]=255
        # label_binary=label_binary.astype('uint8')
        # cv2.imwrite(label_binary_dir+'/'+img_name[0]+'.png',label_binary)

        for i in range(img_h):
            for j in range(img_w):
                cur_change=str(label1[i,j])+'_'+str(label2[i,j])
                #cur_change = str(2) + '_' + str(4)
                change_idx=get_idx(change_type,cur_change)
                label_sta[cur_change].append(change_idx)
                label_new[i,j]=change_idx
        label_new = label_new.astype('uint8')
        cv2.imwrite(label_com_dir + '/' + img_name[0] + '.png', label_new)

    num_total=512*512*len(data_loader)
    total_ratio=0
    for i in range(len(change_type)):
        ratio=len(label_sta[change_type[i]])*1.0/num_total
        total_ratio+=ratio
        print("ratio of %s is %.6f" %(change_type[i],ratio))
    print("total change ratio is %.6f" % total_ratio)








def compute_comb_label0(config,data_loader):
    label_0=[]
    label_1=[]
    label_2 = []
    label3 = []
    label4 = []
    label5 = []
    label6 = []
    label7 = []
    label8 = []
    label9 = []
    label10 = []
    label11 = []
    label12 = []
    label13 = []
    label14 = []
    label15 = []
    img_w,img_h=512,512
    label_binary_dir=config.train_dir+'/label255'
    label_com_dir=config.train_dir+'/label_com'
    mkdir_if_not_exist(label_binary_dir)
    mkdir_if_not_exist(label_com_dir)

    for batch_idx, dataset_dict in enumerate(data_loader, 0):
        print("\rcalculate batch %d" % (batch_idx + 1), end='')
        label12 = dataset_dict['label']
        img_name=dataset_dict['name']
        label1=label12[0,0,...]
        label2 = label12[0, 1, ...]
        label_new=np.zeros_like(label1)
        label_binary=np.zeros_like(label1)
        label_binary[label1>0]=255
        label_binary=label_binary.astype('uint8')
        cv2.imwrite(label_binary_dir+'/'+img_name+'.png',label_binary)
        for i in range(img_h):
            for j in range(img_w):
                if label1[i,j]==0 and label2[i,j]==0:
                    label_0.append(0)
                if label1[i,j]==1 and label2[i,j]==2:
                    label_1.append(1)
                    label_new[i,j]=1
                if label1[i,j]==1 and label2[i,j]==3:
                    label_2.append(2)
                    label_new[i,j]=2
                if label1[i,j]==1 and label2[i,j]==4:
                    label3.append(3)
                    label_new[i,j]=3
                if label1[i,j]==1 and label2[i,j]==5:
                    label4.append(4)
                    label_new[i,j]=4
                if label1[i,j]==1 and label2[i,j]==6:
                    label5.append(5)
                    label_new[i,j]=5

                if label1[i,j]==2 and label2[i,j]==3:
                    label6.append(6)
                    label_new[i,j]=6
                if label1[i,j]==2 and label2[i,j]==4:
                    label7.append(7)
                    label_new[i,j]=7
                if label1[i,j]==2 and label2[i,j]==5:
                    label8.append(8)
                    label_new[i,j]=8
                if label1[i,j]==2 and label2[i,j]==6:
                    label9.append(9)
                    label_new[i,j]=9

                if label1[i,j]==3 and label2[i,j]==4:
                    label10.append(10)
                    label_new[i,j]=10
                if label1[i,j]==3 and label2[i,j]==5:
                    label11.append(11)
                    label_new[i,j]=11
                if label1[i,j]==3 and label2[i,j]==6:
                    label12.append(12)
                    label_new[i,j]=12

                if label1[i,j]==4 and label2[i,j]==5:
                    label13.append(13)
                    label_new[i,j]=13
                if label1[i,j]==4 and label2[i,j]==6:
                    label14.append(14)
                    label_new[i,j]=14

                if label1[i,j]==5 and label2[i,j]==6:
                    label15.append(15)
                    label_new[i,j]=15
        # label_new=label_new.astype('uint8')
        # cv2.imwrite(label_com_dir+'/'+img_name+'.png',label_new)


    num_total=512*512*len(data_loader)
    num0=len(label_0)*1.0/num_total
    num1 = len(label_1) * 1.0 / num_total
    num2 = len(label_2) * 1.0 / num_total
    num3 = len(label3) * 1.0 / num_total
    num4 = len(label4) * 1.0 / num_total
    num5 = len(label5) * 1.0 / num_total
    num6=len(label6)*1.0/num_total
    num7 = len(label7) * 1.0 / num_total
    num8 = len(label8) * 1.0 / num_total
    num9 = len(label9) * 1.0 / num_total
    num10 = len(label10) * 1.0 / num_total
    num11 = len(label11) * 1.0 / num_total
    num12 = len(label12) * 1.0 / num_total
    num13 = len(label13) * 1.0 / num_total
    num14= len(label14) * 1.0 / num_total
    num15 = len(label15) * 1.0 / num_total

    print("ratio of 0_0 is {:.6f},ratio of 1_2 is {:.6f},ratio of 1_4 is {:.6f},ratio of 1_5 is {:.6f}",
          "ratio of 1_6 is {:.6f},ratio of 2_3 is {:.6f},ratio of 0_0 is {:.6f},ratio of 0_0 is {:.6f}"
          "ratio of 0_0 is {:.6f},ratio of 0_0 is {:.6f},ratio of 0_0 is {:.6f},ratio of 0_0 is {:.6f}"
          "ratio of 0_0 is {:.6f},ratio of 0_0 is {:.6f},ratio of 0_0 is {:.6f},ratio of 0_0 is {:.6f}")

