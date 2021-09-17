import math
import os, time,sys,cv2
import numpy as np
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
# import torch modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_loaders.RSCD_dl import RSCD_DL
#from models.utils import Acc
from utils.postprocessing import post_proc
from tqdm import tqdm
from utils.utils import mkdir_if_not_exist
from models.convcrf import convcrf
#from sklearn.metrics import roc_curve,auc,accuracy_score,precision_score,recall_score,f1_score,precision_recall_curve
class Infer(object):
    # init function for class
    def __init__(self, config,net, testDataloader,batchsize=1, cuda=True, gpuID=0
                ):
        dl=RSCD_DL(config)
        self.model_path=dl.config.model_name
        #self.pred_dir=dl.config.pred_dir
        self.batchsize=batchsize#测试时保证imgsize/batchsize=int,使得测试网络时正好遍历完所有测试样本是最理想的情况
        self.batchnum=1
        self.output_size=(1,1,256,256)
        # set the GPU flag
        self.cuda = cuda
        self.gpuID = gpuID
        # define an optimizer
        #self.optimG = torch.optim.Adam(net.parameters(),lr=lr)
        # set the network
        self.net = net
        # set the data loaders
        self.testDataloader = testDataloader
        self.config=dl.config
        self.config.precision_path=dl.config.precision_path
        self.test_dir=dl.config.test_dir
        self.multi_outputs=self.config["network_G_CD"]["multi_outputs"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    def grayTrans(self, img,use_batch=False):
        #img = img.data.cpu().numpy()[0][0]*255.0#equal to img.data.squeeze().cpu().numpy()[0]
        if use_batch==False:
            img = img[0] * 255.0
        else:
            img = img.cpu().numpy()[0][0] * 255.0
        img = (img).astype(np.uint8)
        img = Image.fromarray(img, 'L')
        return img
    def grayTrans_numpy(self, img):
        #img = img.data.cpu().numpy()[0][0]*255.0#equal to img.data.squeeze().cpu().numpy()[0]
        img = img[0] * 255.0
        img = img.astype(np.uint8)
        img = Image.fromarray(img, 'L')
        return img

    def saveCycle_tensor(self,img_tensor):
        img_data=0.5*(img_tensor.squeeze(0)+1.)*255
        img_data = img_data.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(img_data)
        return im


    def predict_x(self,batch_x,net):
        use_3D=False
        with torch.no_grad():
            if use_3D==False:
            # ========================for unet 2D without sigmoid===================
            #   _,output = net.forward(batch_x)  # for unet_2D sigmoid
            #   output=F.sigmoid(output)
            #========================for unet 2D sigmoid,note simoid must be used====================
              # imgs=torch.cat((batch_x,batch_x),dim=1)
              #_,output=net.forward(batch_x)#for multi-unet with fea
              try:
                  output,_,_,_,_,_= net.forward(batch_x)#for unet_2D sigmoid deepsupervison
                  #output = net.forward(batch_x)  # for unet_2D sigmoid
                  #output = torch.argmax(output, dim=1).unsqueeze(1).float()#for softmax
                  #=================for multi-output prabability==================================
                  # output=F.softmax(output,dim=1)# for prabability output
                  # output=output[:,1,:,:].unsqueeze(1).float()# for prabability output
              except RuntimeError as exception:
                  if "out of memory" in str(exception):
                     print("WARNING: out of cuda memory, prediction is now switched using smaller batchsize")
                  if hasattr(torch.cuda, 'empty_cache'):
                     torch.cuda.empty_cache()
                  batch_size = batch_x.size(0)
                  pred_num = int(batch_size / self.batchnum)
                  mod_num = int(batch_size % self.batchnum)

                  output = torch.zeros(size=self.output_size).cuda(0)
                  for i in range(pred_num):
                    temp_out = net.forward(batch_x[i * self.batchnum:(i + 1) * self.batchnum,...])
                    #temp_out = torch.argmax(temp_out, dim=1).unsqueeze(1).float()#for unet_softmax
                    temp_out = temp_out[:, 1, :, :]  # for probability output
                    output = torch.cat((output, temp_out), dim=0)
                  if mod_num > 0:
                    temp_out = net.forward(batch_x[batch_size - mod_num:batch_size, ...])
                    #temp_out = torch.argmax(temp_out, dim=1).unsqueeze(1).float() # for unet_softmax, index output
                    temp_out=temp_out[:,1,:,:]#for probability output
                    output = torch.cat((output, temp_out), dim=0)
                  output = output[1:, ...]
            #==================for unet softmax=====================
            #    pred=net.forward(batch_x)
            # #output = torch.max(pred, dim=1)[0].unsqueeze(1)#返回最大概率值
            #    output=torch.argmax(pred,dim=1).unsqueeze(1)#返回标号值
            #==============for unet 3D=============================
            # output=net.forward(torch.unsqueeze(batch_x,dim=-1))#for unet_3D
            # output = torch.squeeze(output, -1)#for unet_3D
            #=====================for conv3d input=<n,c,d,w,h>================
            else:
               img1 = batch_x[:, 0:3, :, :]
               img2 = batch_x[:, 3:6, :, :]
               imgs_12 = torch.cat((img1.unsqueeze(dim=2), img2.unsqueeze(dim=2)), dim=2)
               pred = net.forward(imgs_12)
               output = torch.argmax(pred, dim=1).unsqueeze(1)  # 返回标号值
               #output=output[:,1,:,:].unsqueeze(1)


        return output
    def predict_xy(self, batch_x, net):
        with torch.no_grad():
            #output = net.forward(batch_x[:, 0:3, ...], batch_x[:, 3:6, ...])
           try:
              output=net.forward(batch_x[:,0:3,...],batch_x[:,3:6,...])
           except RuntimeError as exception:
               if "out of memory" in str(exception):
                   print("WARNING: out of cuda memory, prediction is now switched using smaller batchsize")
                   if hasattr(torch.cuda, 'empty_cache'):
                       torch.cuda.empty_cache()
                   batch_size=batch_x.size(0)
                   pred_num=int(batch_size/self.batchnum)
                   mod_num=int(batch_size%self.batchnum)
                   output=torch.zeros(size=self.output_size).cuda(0)
                   for i in range(pred_num):
                       temp_out=net.forward(batch_x[i*self.batchnum:(i+1)*self.batchnum,0:3,...],batch_x[i*self.batchnum:(i+1)*self.batchnum,3:6,...])
                       output=torch.cat((output,temp_out),dim=0)
                   if mod_num>0:
                       temp_out=net.forward(batch_x[batch_size-mod_num:batch_size,0:3,...],batch_x[batch_size-mod_num:batch_size,3:6,...])
                       output = torch.cat((output, temp_out), dim=0)
                   output=output[1:,...]
                   #====================single by single====================
                   # for i in range(batch_size):
                   #     x1=torch.unsqueeze(batch_x[i, 0:3, ...], dim=0)
                   #     x2=torch.unsqueeze(batch_x[i, 3:6, ...], dim=0)
                   #     temp_out = net.forward(x1, x2)
                   #     temp_out=torch.squeeze(temp_out,dim=0)
                   #     output.append(temp_out.cpu().numpy())
               else:
                   raise exception
        return output
    def predict_img_pad(self,x,target_size,predict,multi_inputs=False):
        '''
                滑动窗口预测图像。
               每次取target_size大小的图像预测，但只取中间的1/4，这样预测可以避免产生接缝。
                :param target_size:
                :return:
                '''
        # target window是正方形，target_size是边长
        #x_gpu=x
        x_cpu=x.cpu().numpy()
        x_cpu=x_cpu.reshape(x_cpu.shape[1],x_cpu.shape[2],x_cpu.shape[3])
        quarter_target_size = target_size // 4
        half_target_size = target_size // 2
        pad_width = (
            (0, 0),
            (quarter_target_size, target_size),#axis=0 填充后+quarter_target_size+target_size
            (quarter_target_size, target_size)#axis=1 填充后+quarter_target_size+target_size
            )
        #pad_x = np.pad(x, pad_width, 'constant', constant_values=0)#（448,784,6）==》（588,924,6）
        pad_x = np.pad(x_cpu, pad_width, 'reflect')
        pad_y = np.zeros(
            (1,pad_x.shape[1], pad_x.shape[2]),
            dtype=np.float32)

        def update_prediction_center(one_batch):
            """根据预测结果更新原图中的一个小窗口，只取预测结果正中间的1/4的区域"""
            wins = []
            for row_begin, row_end, col_begin, col_end in one_batch:
                win = pad_x[:,row_begin:row_end, col_begin:col_end]
                win = np.expand_dims(win, 0)
                wins.append(win)
            x_window = np.concatenate(wins, 0)#(836,256,256,6) for test0
            x_window=torch.from_numpy(x_window).cuda()
            #x_window = torch.from_numpy(x_window)
            if self.config.deep_supervision==True:
               y1, y2, y3, y4 = predict(x_window)
               y_window = y4
            else:
               # if multi_inputs:
               #    y_window=predict(x_window[:,0:3,...],x_window[:,3:6,...])
               # else:
               y_window = predict(x_window)  # 预测一个窗格
               if isinstance(y_window,list):
                  y_window=torch.from_numpy(np.array(y_window)).cuda()

            for k in range(len(wins)):
                row_begin, row_end, col_begin, col_end = one_batch[k]
                pred = y_window[k, ...]
                y_window_center = pred[:,
                                  quarter_target_size:target_size - quarter_target_size,
                                  quarter_target_size:target_size - quarter_target_size
                                  ]  # 只取预测结果中间区域 将正方形四等分，只取中间的1/4区域  pred(112,112,1)==>y_window_center(56,56,1)

                pad_y[:,
                row_begin + quarter_target_size:row_end - quarter_target_size,
                col_begin + quarter_target_size:col_end - quarter_target_size
                 ] = y_window_center.cpu().numpy()  # 更新也，

            # 每次移动半个窗格
        batchs = []
        batch = []
        for row_begin in range(0, pad_x.shape[1], half_target_size):
            for col_begin in range(0, pad_x.shape[2], half_target_size):
                row_end = row_begin + target_size
                col_end = col_begin + target_size
                if row_end <= pad_x.shape[1] and col_end <= pad_x.shape[2]:
                    batch.append((row_begin, row_end, col_begin, col_end))
        if len(batch) > 0:
            batchs.append(batch)
            batch = []
        for bat in tqdm(batchs, desc='Batch pred'):
            update_prediction_center(bat)
        y = pad_y[:,quarter_target_size:quarter_target_size + x_cpu.shape[1],
                quarter_target_size:quarter_target_size + x_cpu.shape[2]
                ]

        return y

    def inferAdv(self,multi_inputs=False,cycleGAN=False):
        self.net.eval()
        #self.model_path0='E:\TEST\DownLoadPrj\GAN\PyTorch-CycleGAN-master\PyTorch-CycleGAN-master\output/netG_B2A.pth'
        cpkt=torch.load(self.model_path,map_location='cpu')
        #self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络
        if cycleGAN:
            self.net.load_state_dict(cpkt['Gis'])
        else:
            self.net.load_state_dict(cpkt)
        image_size=self.config.image_size
        for i, sample in enumerate(self.testDataloader, 0):

            # imgs=sample['image']
            # img_name=sample['name']
            imgs=sample['B']
            img_name=sample['name']

            #=========for conv3D=====================================================

            #========================================================================
            if self.cuda:
                imgs= imgs.cuda(self.gpuID)
                #imgs = imgs_12.cuda(self.gpuID)

            with torch.no_grad():
                if multi_inputs:
                   masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_xy(xx, self.net))#==>numpy
                else:

                   #masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_x(xx, self.net))
                   #=====imgs = torch.unsqueeze(imgs, dim=-1)#for unet_3D not necessary
                   #masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_x(xx, self.net))
                   masks_pred=self.net(imgs)

                im=self.saveCycle_tensor(masks_pred)
                print("processing image {}, size is {}".format(i, masks_pred.shape))
                pred_dir='E:\TestData\Air-CD\SZTAKI_AirChange_Benchmark\SZTAKI_AirChange_Benchmark/train/raw/aug0\T2_1_cycle=multi_unetnormaldis2'
                im.save('%s/%s.png'% (pred_dir,img_name[0]))

                #print("processing image size is:",masks_pred.shape)
                # print("processing image {}, size is {}".format(i,masks_pred.shape))
                # #masks_pred=self.net(imgs)#输入必须是训练集的整数倍
                # predict_img=self.grayTrans_numpy(masks_pred)
                # predict_img.save('%s/%s.png'% (self.pred_dir,img_name[0]))
                # predict_img = np.array(predict_img).astype('uint8')
                # _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)
                # cv2.imwrite('%s/%s%s.png'% (self.pred_dir,'/Binary/',img_name), binary_img)
                # # =======remove hole===============
                # save_dir = self.pred_dir + '/remove_hole_area'
                # mkdir_if_not_exist(save_dir)
                # res_img = post_proc(binary_img)
                # cv2.imwrite(save_dir + '/' + img_name[0] + '.tif', res_img)
                # # =======res img 1 nochange 2 change
                # res_img = np.array(res_img).astype('float')
                # res_img /= 255
                # res_img += 1
                # res_img = np.array(res_img).astype('uint8')
                # save_dir = self.pred_dir + '/Upload'
                # mkdir_if_not_exist(save_dir)
                # cv2.imwrite(save_dir + '/' + img_name[0] + '.tif', res_img)

    def draw_roc_curve(self):

        y_true_path=self.pred_dir+'/ground_truth.pnz.npy'
        y_pred_path = self.pred_dir + '/prediction.pnz.npy'


        y_true=np.load(y_true_path)
        y_pred=np.load(y_pred_path)

        fpr, tpr, _ = roc_curve(y_true, y_pred)  ##y_pred is probability

        auc1 = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def compute_pred_evaluation_ave(self):


        start_time = time.clock()


        #y_true, y_pred =self.inferSen(multi_inputs=False)
        precision, recall, f1_score, acc = self.inferSenBR(use_ave=True)


        end_time = time.clock()
        run_time = (end_time - start_time)
        mylog = open(self.config.precison_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        print("model is %s" % self.config.model_name)
        print("prediction time is %fs" % run_time)

        print("precision is %.4f" % precision)
        print("recall is %.4f" % recall)
        print("f1_score is %.4f" % f1_score)
        print("overall accuracy is %.4f" % acc)

        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup

    def DeNormalization(self,rgb_mean, im_tensor, min_max=(0, 1), max255=False):

        if max255 == False:
            im_tensor = im_tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
            im_tensor = (im_tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            im = im_tensor.numpy().transpose([1, 2, 0])
            im = (im + rgb_mean) * 255
        else:
            min_max = (0, 255)
            im_tensor = im_tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
            # im_tensor = (im_tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            im = im_tensor.numpy().transpose([1, 2, 0])

        im = im.astype('uint8')
        return Image.fromarray(im, 'RGB')
    def inferStyle(self):
        start_time = time.clock()
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络

        # style_img_path=r'E:\TestData\Building&Road\Massachusett_Building\trainNew\patch512\train80\aug7\train\img\4.jpg'
        # style_img=cv2.imread(style_img_path,cv2.IMREAD_UNCHANGED)
        # style_img=torch.from_numpy(style_img.transpose((2,0,1)))
        # style_img = style_img.float().div(255).unsqueeze(0).cuda(0)

        meanB = np.load(self.config.meanB)
        stdB = np.load(self.config.stdB)
        meanB = torch.from_numpy(meanB).float().to(self.device)
        stdB = torch.from_numpy(stdB).float().to(self.device)

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):

            imgs=sample['img']
            img_name=sample['name']#img_name[0]

            if self.cuda:
                imgs= imgs.cuda(self.gpuID)
                imgs=imgs.unsqueeze(0)#for batch_size=1

            with torch.no_grad():
                masks_pred=self.net(imgs,meanB,stdB)

                #masks_pred = masks_pred[0].data.cpu().numpy()
                rgb_mean = [0, 0, 0]
                predict_img = self.DeNormalization(rgb_mean, masks_pred, max255=False)
                predict_img.save('%s/%s.png' % (self.pred_dir, img_name))




        end_time = time.clock()
        run_time = (end_time - start_time)
        mylog = open(self.config.precison_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        print("model is %s" % self.config.model_name)
        print("prediction time is %fs" % run_time)

    def compute_pred_evaluation(self,use_TTA=False,use_scaleATT=False):



        start_time = time.clock()


        y_true, y_pred, y_pred_p = self.inferSenBR(multi_outputs=self.multi_outputs,use_TTA=use_TTA,use_scaleATT=use_scaleATT)
        # ========================================
        img_num=y_true.shape[0]
        end_time = time.clock()
        run_time = (end_time - start_time)
        mylog = open(self.config.precison_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        print("model is %s" % self.config.model_name)
        print("prediction time is %fs" % run_time)

        precision=0.0
        recall=0.0
        acc=0
        f1=0.0
        test_num=0
        precision, recall, f1_score_value,acc,kappa,iou_score = self.PR_score_whole(y_true, y_pred)


        print("precision is %.4f" % precision)
        print("recall is %.4f" % recall)
        print("f1_score is %.4f" % f1_score_value)
        print("overall accuracy is %.4f" % acc)
        print("kappa is %.4f" % kappa)
        print("iou_score is %.4f" % iou_score)


        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup
        return  f1_score_value,iou_score


    def CD_Evaluation_Sense(self,use_score=False,use_TTA=False,use_model_ensemble=False,output_rgb=False,use_con=False,use_CRF=False,mode='_best_loss'):

        start_time = time.clock()
        if use_score:
            if use_model_ensemble:
                IoU_mean, Sek, Score = self.inferSenCD2_score_label7_TTA_BT_Com(use_score=use_score, output_rgb=output_rgb, mode=mode,
                                                                 use_model_ensemble=use_model_ensemble)
            else:
                IoU_mean, Sek, Score = self.inferSenCD2_score_label7_TTA_BT(use_TTA=use_TTA,use_score=use_score, output_rgb=output_rgb, mode=mode,
                                                                                                        use_model_ensemble=use_model_ensemble,use_con=use_con,use_CRF=use_CRF)




        else:
            if use_model_ensemble:
                self.inferSenCD2_score_label7_TTA_BT_Com(use_score=use_score, output_rgb=output_rgb, mode=mode,
                                                         use_model_ensemble=use_model_ensemble)
            else:
                self.inferSenCD2_score_label7_TTA_BT(use_TTA=use_TTA,use_score=use_score, output_rgb=output_rgb, mode=mode,
                                                         use_model_ensemble=use_model_ensemble)



        end_time = time.clock()
        run_time = (end_time - start_time)
        #save_presion_path=self.config.pred1_dir + '\\precison'+mode+'.txt'

        mylog = open(self.config.precision_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        print("model is %s" % self.config.model_name)
        print("prediction time is %fs" % run_time)
        if use_score:
            print("prediction mIOU is %.6f" % IoU_mean)
            print("prediction Sek is %.6f" % Sek)
            print("prediction Score is %.6f" % Score)

        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup


    def inferSenCD2(self,use_TTA=False,mode='_best_acc'):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        '''

        self.net.eval()
        mode_snap = mode
        model_path = self.config.model_dir + '/' + self.config.pred_name + mode_snap + '.pth'
        self.net.load_state_dict(torch.load(model_path))
        #self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络
        image_size=self.config.patch_size
        target_test=[]
        pred_test=[]


        precision = 0.0
        recall = 0.0
        acc = 0
        f1_score= 0.0
        # #pred_batch = 16
        # pred_batch=51
        # test_num = len(self.testDataloader)/pred_batch
        # test_mod=len(self.testDataloader)%pred_batch
        # test_batch=pred_batch*test_num

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):

            imgs=sample['img']
            img_name=sample['name']#img_name[0]
            label_test=sample['label']

            label_test = label_test[0].squeeze(0).data.numpy()#[512,512]
            #label_test=label_test.squeeze(0).data.numpy()#[1,512,512]
            if self.cuda:
                imgs= imgs.cuda(self.gpuID)#[1,6,512,512]


            with torch.no_grad():
                if use_TTA:
                    imgs_aug=[]
                    masks_pred=[]
                    imgs = imgs.data.cpu().numpy()
                    imgs_aug = [imgs,
                                np.rot90(imgs, 1, (2, 3)), np.rot90(imgs, 2, (2, 3)), np.rot90(imgs, 3, (2, 3))
                                # imgs[:, :, ::-1, :],imgs[:, :, :, ::-1],imgs[:, :, ::-1, ::-1]
                                ]
                    cur_img = torch.from_numpy(np.array(imgs_aug).copy()).squeeze(
                        1).cuda()  # must use .copy() after flip operation
                    if self.config["network_G_CD"]["which_model_G"] == "EDCls_UNet_BCD_WHU" or \
                            self.config["network_G_CD"]["which_model_G"] == "EDCls_UNet_BCD_WHU_DCN" or \
                            self.config["network_G_CD"]["which_model_G"] == "EDCls_UNet_BCD_WHU_DCN2" or \
                            self.config["network_G_CD"]["which_model_G"] == "EDCls_UNet_BCD_WHU_STN" or \
                            self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_WHU_STN2':
                        if self.config["network_G_CD"]["use_DS"]:
                            _, _, _, preds = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        else:
                            preds = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])


                    else:
                        preds = self.net(cur_img)  # for FC-EF
                    preds=preds.data.cpu().numpy()
                    for i in range(len(imgs_aug)):
                        if i == 0:
                            cur_pred=preds[i]
                        if i == 1:
                            cur_pred = np.rot90(preds[i].copy(), -1, (1, 2))

                        if i == 2:
                            cur_pred = np.rot90(preds[i].copy(), -2, (1, 2))

                        if i == 3:
                            cur_pred = np.rot90(preds[i].copy(), -3, (1, 2))

                        # if i == 4:
                        #     pred1 = preds1[i].copy()[:, ::-1, :]
                        #     pred2 = preds2[i].copy()[:, ::-1, :]
                        # if i == 5:
                        #     pred1 = preds1[i].copy()[:, :, ::-1]
                        #     pred2 = preds2[i].copy()[:, :, ::-1]
                        # if i == 6:
                        #     pred1 = preds1[i].copy()[:, ::-1, ::-1]
                        #     pred2 = preds2[i].copy()[:, ::-1, ::-1]

                        masks_pred.append(cur_pred)


                    # _preds1 = np.average(preds1_mask, axis=0)  # [7,512,512]
                    # _preds2 = np.average(preds2_mask, axis=0)
                    masks_pred=np.average(masks_pred,axis=0)




                else:
                    if self.config["network_G_CD"]["which_model_G"] == "EDCls_UNet_BCD_WHU" or \
                            self.config["network_G_CD"]["which_model_G"] == "EDCls_UNet_BCD_WHU_DCN" or \
                            self.config["network_G_CD"]["which_model_G"] == "EDCls_UNet_BCD_WHU_DCN2" or \
                            self.config["network_G_CD"]["which_model_G"] == "EDCls_UNet_BCD_WHU_STN" or \
                            self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_WHU_STN2':
                        if self.config["network_G_CD"]["use_DS"]:
                            _, _, _, masks_pred = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                        else:
                            masks_pred = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        masks_pred = self.net(imgs)  # for FC-EF
                    masks_pred = masks_pred[0].data.cpu().numpy()




                #print("processing image size is:",masks_pred.shape)
                print("processing image {}, size is {}".format(i,masks_pred.shape))
                #masks_pred=self.net(imgs)#输入必须是训练集的整数倍
                predict_img=self.grayTrans_numpy(masks_pred)
                predict_img.save('%s/%s.png'% (self.config.pred_dir,img_name[0]))
                #pred_test_p+=[masks_pred[0]]
                predict_img = np.array(predict_img).astype('uint8')
                _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)#set thresh=90 to fill the fake boundary caused by inaccurate labeling
                cv2.imwrite('%s/%s%s.png'% (self.config.pred_dir,'/Binary/',img_name[0]), binary_img)
                # =======remove hole===============
                save_dir = self.config.pred_dir + '/remove_hole_area'
                mkdir_if_not_exist(save_dir)
                res_img = post_proc(binary_img)
                cv2.imwrite(save_dir + '/' + img_name[0] + '.png', res_img)

                res_img = np.array(res_img).astype('float')
                res_img /= 255#=====================for acc evaluation, y_pred must be [0,1]
                #pred_test+=[res_img]#=============for accuracy evaluation

                binary_img = np.array(binary_img).astype('float')  # for raw binary
                binary_img /= 255
                #
                pred_test += [res_img]
                target_test += [label_test]


        return  np.array(target_test),np.array(pred_test)


    def CD_Evaluation(self,use_TTA=False):

        start_time = time.clock()

        y_true, y_pred= self.inferSenCD2(use_TTA=use_TTA)
        # ========================================
        img_num = y_true.shape[0]
        end_time = time.clock()
        run_time = (end_time - start_time)
        mylog = open(self.config.precision_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        print("model is %s" % self.config.model_name)
        print("prediction time is %fs" % run_time)

        precision = 0.0
        recall = 0.0
        acc = 0
        f1 = 0.0
        test_num = 0
        precision, recall, f1_score_value, acc, kappa, iou_score = self.PR_score_whole(y_true, y_pred)

        print("precision is %.4f" % precision)
        print("recall is %.4f" % recall)
        print("f1_score is %.4f" % f1_score_value)
        print("overall accuracy is %.4f" % acc)
        print("kappa is %.4f" % kappa)
        print("iou_score is %.4f" % iou_score)

        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup
        return f1_score_value, acc,kappa











    def compute_pred_evaluation_fromImg(self,pred_dir):

            start_time = time.clock()
            #y_true, y_pred, y_pred_p = self.inferSen1()
            gt_dir=self.test_dir+'/label'
            binary_dir=pred_dir+'/Binary'
            img_list=os.listdir(pred_dir+'/Binary')
            y_true=[]
            y_pred=[]

            for i in range (len(img_list)):
                pred_img_path=binary_dir+'/'+img_list[i]
                substr = '.'
                index1 =img_list[i].rfind(substr, 0, len(img_list[i]))
                img_name=img_list[i][0:index1]
                #gt_img_path=gt_dir+'/'+img_name+'.jpg'
                try:
                    label_test = np.asarray(Image.open(gt_dir+'/'+img_name+'.jpg').convert('L'))
                except:

                    label_test = np.asarray(Image.open(gt_dir+'/'+img_name+'.tif').convert('L'))
                pred_test=np.asarray(Image.open(pred_img_path).convert('L'))
                #label_test = np.asarray(Image.open(gt_img_path).convert('L'))
                pred_test = pred_test.astype('float')
                pred_test /= 255
                label_test = label_test.astype('float')
                label_test /= 255
                y_pred+=[pred_test]
                y_true+=[label_test]
            # ========================================
            y_pred=np.array(y_pred)
            y_true=np.array(y_true)
            img_num = y_true.shape[0]
            end_time = time.clock()
            run_time = (end_time - start_time)
            mylog = open(pred_dir+'/precision_new.txt', 'w')
            stdout_backup = sys.stdout
            sys.stdout = mylog


            precision = 0.0
            recall = 0.0
            acc = 0
            f1 = 0.0
            test_num = 0
            precision, recall, f1_score_value, acc, kappa = self.PR_score_whole(y_true, y_pred)


            print("precision is %.4f" % precision)
            print("recall is %.4f" % recall)
            print("f1_score is %.4f" % f1_score_value)
            print("overall accuracy is %.4f" % acc)
            print("kappa is %.4f" % kappa)

            print('Finish!')
            mylog.close()
            sys.stdout = stdout_backup


    def f1_score(self,truth, preds, smooth=1):
        """
        所有图像一同计算f1，而不是单独计算。
        """
        inter = 0
        ps = 0
        ts = 0
        for t, p in zip(truth, preds):
            tr = np.ravel(t)
            pr = np.ravel(p)
            inter += np.sum(tr * pr)
            ps += np.sum(pr)
            ts += np.sum(tr)
        f1 = (2 * inter + smooth) / (ps + ts + smooth)
        return f1

    def PR_score_whole(self, y_true, y_pred):
        smooth=1.0
        # y_true = y_true[:, :]#still 2-dim vector
        # y_pred = y_pred[:, :]


        y_true=np.ravel(y_true)#1-dim vector [1825,256,256]==>[119603200]
        y_pred=np.ravel(y_pred)#1-dim vector [1825,256,256]==>[119603200]

        # inter = 0
        # ps = 0
        # ts = 0
        # for t, p in zip(truth, preds):
        #     tr = np.ravel(t)
        #     pr = np.ravel(p)
        #     inter += np.sum(tr * pr)
        #     ps += np.sum(pr)
        #     ts += np.sum(tr)
        # f1 = (2 * inter + smooth) / (ps + ts + smooth)
        # return f1


        c1 = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))#the clip opetration make sure y_pred y_true and y_true * y_pred are [0,1]
        c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
        c3 = np.sum(np.round(np.clip(y_true, 0, 1)))
        #=======================================================

        # How many selected items are relevant?
        precision = c1*1.0/ c2
        # How many relevant items are selected?
        recall = c1*1.0/ c3
        # Calculate f1_score
        f1_score = (2 * precision * recall) / (precision + recall)#
        iou_score=(c1)/(c2+c3-c1)


        y_true=np.array(y_true)
        y_pred = np.array(y_pred).astype('float')#must be float
        cond1=(y_true == 1)#119603200 [True,False,...]
        cond2=(y_pred>0.5)
        cond3=(y_true == 0)
        cond4=(y_pred<0.5)
        idx_TP = np.where(cond1&cond2)[0]#not np.where( a and b) np.where(cond1&cond2)==>tuple
        idx_FP = np.where(cond3&cond2)[0]
        idx_FN=np.where(cond1&cond4)[0]
        idx_TN=np.where(cond3&cond4)[0]
        #pix_number = (y_pred.shape[0] * y_pred.shape[1])
        pix_number=y_pred.shape[0]
        acc=(len(idx_TP)+len(idx_TN))*1.0/pix_number

        nTPNum=len(idx_TP)
        nFPNum=len(idx_FP)
        nFNNum=len(idx_FN)
        nTNNum=len(idx_TN)
        temp1 = ((nTPNum + nFPNum) / 1e5) * ((nTPNum + nFNNum) / 1e5)
        temp2 = ((nFNNum + nTNNum) / 1e5) * ((nFPNum + nTNNum) / 1e5)
        temp3 = (pix_number / 1e5) * (pix_number / 1e5)
        dPre = (temp1 + temp2) * 1.0 / temp3
        kappa = (acc - dPre) / (1 - dPre)

        '''
         import numpy as np
 from sklearn.metrics import roc_curve
 y = np.array([1,1,2,2])
 pred = np.array([0.1,0.4,0.35,0.8])
 fpr, tpr, thresholds = roc_curve(y, pred)
 print(fpr)
 print(tpr)
 print(thresholds)
 plt.figure(1)
 plt.plot(fpr, tpr)
 plt.show()
 from sklearn.metrics import auc
 print(auc(fpr, tpr))
         '''




        return  precision, recall, f1_score,acc,kappa,iou_score



    def PR_score(self, y_true, y_pred):

        y_true = y_true[:, :]
        y_pred = y_pred[:, :]

        c1 = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))#the clip opetration make sure y_pred y_true and y_true * y_pred are [0,1]
        c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
        c3 = np.sum(np.round(np.clip(y_true, 0, 1)))
        image_black=False
        # If there are no true samples, fix the F1 score at 0.
        if c3==0 or c2==0:
            image_black=True
            return 0,0,0,image_black,0
        if c1==0:
            return 0,0,0,image_black,0
        #===============to make the result more reasonable, c2==0 and c3!=0, image_black=False
        # if c3 == 0:
        #     image_black = True
        #     return 0, 0, 0, image_black, 0
        # else:
        #     if c1 == 0 or c2 == 0:
        #         image_black = False
        #         return 0, 0, 0, image_black, 0

        # How many selected items are relevant?
        precision = c1*1.0/ c2
        # How many relevant items are selected?
        recall = c1*1.0/ c3
        # Calculate f1_score
        f1_score = 2 * (precision * recall) / (precision + recall)

        y_true=np.array(y_true)
        y_pred = np.array(y_pred).astype('float')#must be float
        cond1=(y_true == 1)
        cond2=(y_pred>0.5)
        cond3=(y_true == 0)
        cond4=(y_pred<0.5)
        idx_TP = np.where(cond1&cond2)[0]#not np.where( a and b)
        #idx_FP = np.where(cond3&cond2)[0]
        #idx_FN=np.where(cond1&cond4)[0]
        idx_TN=np.where(cond3&cond4)[0]
        pix_number=(y_pred.shape[0]*y_pred.shape[1])
        acc=(len(idx_TP)+len(idx_TN))*1.0/pix_number
        # temp1=(len(idx_TP)+len(idx_FP))*1.0/(len(idx_TP)+len(idx_FN))
        # temp2 = (len(idx_TN) + len(idx_FN)) * 1.0 / (len(idx_FP) + len(idx_TN))
        # temp3=(len(idx_TP)+len(idx_TN)+len(idx_FN)+len(idx_FP))*(len(idx_TP)+len(idx_TN)+len(idx_FN)+len(idx_FP))
        # dPre = (temp1 + temp2) * 1.0 / temp3
        # dKappa = (acc - dPre) / (1 - dPre)
        return  precision, recall, f1_score,image_black,acc

    def inferSen1(self,multi_inputs=False):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        '''
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络
        image_size=self.config.image_size
        target_test=[]
        pred_test=[]
        pred_test_p=[]

        for i, sample in enumerate(self.testDataloader, 0):
             #if multi_inputs==False:
            imgs=sample['image']
            img_name=sample['name']
            label_path = self.test_dir + '/label/' + img_name[0]+".jpg"
            label_path1 = self.test_dir + '/label/' + img_name[0] + ".bmp"
            label_path2 = self.test_dir + '/label/' + img_name[0] + ".tif"
            try:
                 label_test = np.asarray(Image.open(label_path).convert('L'))
            except:
                try:
                    label_test = np.asarray(Image.open(label_path1).convert('L'))
                except:
                    label_test = np.asarray(Image.open(label_path2).convert('L'))


            # except:
            #      label_test = np.asarray(Image.open(label_path2).convert('L'))
            #label_test= np.asarray(Image.open(label_path).convert('L'))
            label_test =label_test.astype('float')
            label_test /= 255
            target_test+=[label_test]
            if self.cuda:
                imgs= imgs.cuda(self.gpuID)

            with torch.no_grad():
                if multi_inputs:
                   masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_xy(xx, self.net))#==>numpy
                else:

                   #masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_x(xx, self.net))
                   #=====imgs = torch.unsqueeze(imgs, dim=-1)#for unet_3D not necessary
                   masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_x(xx, self.net))


                #print("processing image size is:",masks_pred.shape)
                print("processing image {}, size is {}".format(i,masks_pred.shape))
                #masks_pred=self.net(imgs)#输入必须是训练集的整数倍
                predict_img=self.grayTrans_numpy(masks_pred)
                predict_img.save('%s/%s.png'% (self.pred_dir,img_name[0]))
                pred_test_p+=[masks_pred[0]]
                predict_img = np.array(predict_img).astype('uint8')
                _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)#set thresh=90 to fill the fake boundary caused by inaccurate labeling
                cv2.imwrite('%s/%s%s.png'% (self.pred_dir,'/Binary/',img_name[0]), binary_img)
                # =======remove hole===============
                save_dir = self.pred_dir + '/remove_hole_area'
                mkdir_if_not_exist(save_dir)
                res_img = post_proc(binary_img)
                cv2.imwrite(save_dir + '/' + img_name[0] + '.png', res_img)

                res_img = np.array(res_img).astype('float')
                res_img /= 255#=====================for acc evaluation, y_pred must be [0,1]
                #pred_test+=[res_img]#=============for accuracy evaluation

                binary_img = np.array(binary_img).astype('float')  # for raw binary
                binary_img /= 255
                pred_test += [binary_img]
                # ==============================res img 1 nochange 2 change========================
                # res_img = np.array(res_img).astype('float')
                # res_img /= 255
                # res_img += 1
                # res_img = np.array(res_img).astype('uint8')
                # save_dir = self.pred_dir + '/Upload'
                # mkdir_if_not_exist(save_dir)
                # cv2.imwrite(save_dir + '/' + img_name[0] + '.tif', res_img)
        return  np.array(target_test),np.array(pred_test),np.array(pred_test_p)

    def predict_TTA(self, net,inputs, multi_outputs=False):
        '''
        :param net:
        :param inputs:
        :return: avg outputs after TTA
        '''
        # import ttach as tta
        # transforms = tta.Compose(
        #     [
        #         #tta.HorizontalFlip(),
        #         #tta.Rotate90(angles=[0,180]),
        #         tta.Scale(scales=[0.5,1, 2]),
        #         #tta.Multiply(factors=[0.9, 1, 1.1]),
        #     ]
        # )
        #
        # tta_model = tta.SegmentationTTAWrapper(net, transforms)
        # #tta_model = tta.SegmentationTTAWrapper(net, tta.aliases.d4_transform(), merge_mode='mean')
        # if multi_outputs:
        #     output=None
        #     trans_num=0
        #     for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
        #         trans_num+=1
        #         # augment image
        #         augmented_image = transformer.augment_image(inputs)
        #
        #         # pass to model
        #         model_output = net(augmented_image)
        #
        #         # reverse augmentation for mask and label
        #         deaug_mask = transformer.deaugment_mask(model_output['mask'])
        #
        #
        #         # save results
        #         output+=deaug_mask
        #
        #
        #     # reduce results as you want, e.g mean/max/min
        #     output=output/trans_num
        #
        #
        # else:
        #     output=tta_model(inputs)
        # return output



        # if multi_outputs:
        #     _, output0, _ = net(inputs)
        #
        #     output1 = torch.flip(net(torch.flip(inputs, [2]))[1], [2])
        #     output2 = torch.flip(net(torch.flip(inputs, [3]))[1], [3])
        #     output3 = torch.rot90(net(torch.rot90(inputs, 1, [2, 3]))[1], -1, [2, 3])
        #     output4 = torch.rot90(net(torch.rot90(inputs, 2, [2, 3]))[1], -2, [2, 3])
        #     output5 = torch.rot90(net(torch.rot90(inputs, 3, [2, 3]))[1], -3, [2, 3])
        # else:
        #     output0 = net(inputs)
        #     output1 = torch.flip(net(torch.flip(inputs, [2])), [2])
        #     output2 = torch.flip(net(torch.flip(inputs, [3])), [3])
        #     output3 = torch.rot90(net(torch.rot90(inputs, 1, [2, 3])), -1, [2, 3])
        #     output4 = torch.rot90(net(torch.rot90(inputs, 2, [2, 3])), -2, [2, 3])
        #     output5 = torch.rot90(net(torch.rot90(inputs, 3, [2, 3])), -3, [2, 3])
        #
        # return (output0 + output1 + output2 + output3 + output4 + output5) / 6
        #===============================for scale TTA=====================
        output=net(inputs)
        inputs0=F.interpolate(inputs,scale_factor=0.5,mode='bilinear',align_corners=True)
        output0=F.interpolate(net(inputs0),scale_factor=2,mode='bilinear',align_corners=True)

        inputs1 = F.interpolate(inputs, scale_factor=1.5, mode='bilinear', align_corners=True)
        output1 = F.interpolate(net(inputs1), size=inputs.size()[2:], mode='bilinear', align_corners=True)

        inputs2 = F.interpolate(inputs, scale_factor=2, mode='bilinear', align_corners=True)
        output2 = F.interpolate(net(inputs2), scale_factor=0.5, mode='bilinear', align_corners=True)

        return (output0 + output1 +output2+ output) / 4

     #
     # def generate_rgb_table(self,change_label):


    # def inferSenCD2(self):
    #     '''
    #     output y_true,y_pred, y_pred_p(of probability not 0,1)
    #     :return:
    #     '''
    #
    #     self.net.eval()
    #     self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络
    #
    #
    #     change_type = ['0_0',
    #                    '1_2', '1_3', '1_4', '1_5', '1_6',
    #                    '2_1', '2_3', '2_4', '2_5', '2_6',
    #                    '3_1', '3_2', '3_4', '3_5', '3_6',
    #                    '4_1', '4_2', '4_3', '4_5', '4_6',
    #                    '5_1', '5_2', '5_3', '5_4', '5_5', '5_6',
    #                    '6_1', '6_2', '6_3', '6_4', '6_5']
    #     rgb_table={'0':(255,255,255),'1':(0,0,255),'2':(128,128,128),'3':(0,128,0),
    #                '4':(0,255,0),'5':(128,0,0),'6':(255,0,0)}
    #
    #
    #
    #     for i, sample in enumerate(tqdm(self.testDataloader, 0)):
    #
    #         imgs=sample['img']
    #         img_name=sample['name']#img_name[0]
    #         #label_test=sample['label']
    #
    #         #label_test=label_test.squeeze(0).data.numpy()
    #         if self.cuda:
    #             imgs= imgs.cuda(self.gpuID)
    #             #imgs=imgs.unsqueeze(0)#for batch_size=1
    #         print("processing image {}, size is {}".format(i, imgs.shape))
    #         with torch.no_grad():
    #
    #             masks_pred = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
    #             #masks_pred=F.softmax()
    #             masks_pred = torch.argmax(masks_pred, dim=1)
    #             masks_pred = masks_pred[0].data.cpu().numpy().astype('uint8')
    #             #
    #             pred1=np.zeros((masks_pred.shape+(3,)),dtype='uint8')
    #             pred2=np.zeros((masks_pred.shape+(3,)),dtype='uint8')
    #
    #             for i in range(masks_pred.shape[0]):
    #                 for j in range(masks_pred.shape[1]):
    #                     cur_change=change_type[masks_pred[i,j]]
    #                     idx1 = cur_change[:cur_change.find('_')]
    #                     idx2 = cur_change[cur_change.find('_') + 1:]
    #                     key1=str(idx1)
    #                     key2=str(idx2)
    #                     for k in range (3):
    #                         pred1[i,j,2-k]=rgb_table[key1][k]# opencv should write in BGR mode
    #                         pred2[i, j, 2-k] = rgb_table[key2][k]
    #
    #                     # pred1[i,j]=idx1
    #                     # pred2[i,j]=idx2
    #
    #         cv2.imwrite(self.config.pred1_dir+'/'+img_name[0]+'.png',pred1)
    #         cv2.imwrite(self.config.pred2_dir +'/'+ img_name[0] + '.png', pred2)

    def inferSenCD2_label7(self,use_TTA=False):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        '''

        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络

        rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                     '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):

            imgs = sample['img']
            img_name = sample['name']  # img_name[0]
            # label_test=sample['label']

            # label_test=label_test.squeeze(0).data.numpy()
            if self.cuda:
                imgs = imgs.cuda(self.gpuID)
                # imgs=imgs.unsqueeze(0)#for batch_size=1
            print("processing image {}, size is {}".format(i, imgs.shape))
            with torch.no_grad():

                masks_pred1,masks_pred2 = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                # masks_pred=F.softmax()
                masks_pred1 = torch.argmax(masks_pred1, dim=1)
                masks_pred1 = masks_pred1[0].data.cpu().numpy().astype('uint8')
                masks_pred2 = torch.argmax(masks_pred2, dim=1)
                masks_pred2 = masks_pred2[0].data.cpu().numpy().astype('uint8')
                #
                pred1 = np.zeros((masks_pred1.shape + (3,)), dtype='uint8')
                pred2 = np.zeros((masks_pred2.shape + (3,)), dtype='uint8')

                for i in range(masks_pred1.shape[0]):
                    for j in range(masks_pred1.shape[1]):

                        key1 = str(masks_pred1[i,j])
                        key2 = str(masks_pred2[i,j])
                        for k in range(3):
                            pred1[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                            pred2[i, j, 2 - k] = rgb_table[key2][k]



            cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', pred1)
            cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', pred2)


    def model_para_ensemble(self):#cannot use for model generated during different training
        import models.Satt_CD.networks as networks
        #model = networks.define_G_CD(config).to(device)
        import collections
        model1_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc.pth'
        model2_path = self.config.model_dir + '/' + self.config.pred_name + '_best_loss.pth'
        model3_path = self.config.model_dir + '/' + self.config.pred_name + '_final_iter.pth'

        models=[]
        model1=networks.define_G_CD(self.config).to(self.device)#must define model instead of using self.net
        model2 = networks.define_G_CD(self.config).to(self.device)
        model3 = networks.define_G_CD(self.config).to(self.device)

        model1.load_state_dict(torch.load(model1_path))
        model2.load_state_dict(torch.load(model2_path))
        model3.load_state_dict(torch.load(model3_path))

        models.append(model1)
        models.append(model2)
        models.append(model3)

        worker_state_dict = [x.state_dict() for x in models]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()

        # for key in weight_keys:
        #     key_sum = 0
        #     for i in range(len(models)):
        #         key_sum = key_sum + worker_state_dict[i][key]
        #
        #     fed_state_dict[key] = key_sum / len(models)

        # for key in weight_keys:
        #     key_sum = 0
        #     for i in range(len(models)):
        #
        #         if i==0:
        #             key_sum+=worker_state_dict[i][key]*0.6
        #         else:
        #             key_sum += worker_state_dict[i][key] * 0.4#model weight is better than average
        #     fed_state_dict[key] = key_sum

        for key in weight_keys:
            key_sum = 0
            for i in range(len(models)):
                #key_sum = key_sum + worker_state_dict[i][key]
                if i==0:
                    key_sum+=worker_state_dict[i][key]*0.5
                elif i==1:
                    key_sum += worker_state_dict[i][key] * 0.2
                else:
                    key_sum += worker_state_dict[i][key] * 0.3#model weight is better than average


            fed_state_dict[key] = key_sum
        #============update fed weights to fl model============
        self.net.load_state_dict(fed_state_dict)


    def model_out_ensemble(self,output_rgb=False,output_score=False):#cannot use for model generated during different training
        import models.Satt_CD.networks as networks
        model_list=[]
        dict_list=[]

        model0_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc_c0.pth'
        model1_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc_c1.pth'
        model2_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc_c2.pth'

        model0=networks.define_G_CD(self.config).to(self.device)#must define model instead of using self.net
        model1 = networks.define_G_CD(self.config).to(self.device)
        model2 = networks.define_G_CD(self.config).to(self.device)

        model0.load_state_dict(torch.load(model0_path))
        model1.load_state_dict(torch.load(model1_path))
        model2.load_state_dict(torch.load(model2_path))

        model_list.append(model0)
        model_list.append(model1)
        model_list.append(model2)

        rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                     '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}

        infer_list = []
        label_list = []

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):
            with torch.no_grad():
                img_name = sample['name']
                imgs, labels = sample['img'], sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                print("processing image {}, size is {}".format(i, imgs.shape))
                if output_score:
                    gt1_label = labels[0, 0, :, :].data.cpu().numpy().astype('uint8')  # [1,512,512]
                    gt2_label = labels[0, 1, :, :].data.cpu().numpy().astype('uint8')

                imgs_aug = []
                imgs = imgs.data.cpu().numpy()
                imgs_aug.append(imgs)
                # imgs_aug.append(imgs[:,:,::-1,:])
                # imgs_aug.append(imgs[:, :, :, ::-1])
                # imgs_aug.append(imgs[:, :, ::-1, ::-1])

                imgs_aug.append(np.rot90(imgs, 1, (2, 3)))
                imgs_aug.append(np.rot90(imgs, 2, (2, 3)))
                imgs_aug.append(np.rot90(imgs, 3, (2, 3)))

                preds1_mask = []
                preds2_mask = []
                for i in range(len(imgs_aug)):
                    cur_img = torch.from_numpy(imgs_aug[i].copy()).cuda()  # must use .copy() after flip operation
                    if self.config["train"]["use_label_rgb255"]:
                        _, _, _, (logits1, logits2, _) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                    elif self.config["train"]["use_CatOut"]:
                        _, _, _, _, (logits1, logits2) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                    else:
                        #_, _, _, (logits1, logits2) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        _,_,_,(logits1_0,logits2_0)=model0(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        _, _, _, (logits1_1, logits2_1) = model1(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        _, _, _, (logits1_2, logits2_2) = model2(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        # logits1=(logits1_0+logits1_1+logits1_2)/3.0
                        # logits2 = (logits2_0 + logits2_1 + logits2_2) / 3.0
                    #output = torch.mean(torch.cat(output_list), 0).squeeze()
                    preds1_0,preds1_1,preds1_2=torch.softmax(logits1_0, dim=1),torch.softmax(logits1_1, dim=1),\
                                               torch.softmax(logits1_2, dim=1)
                    preds2_0, preds2_1, preds2_2 = torch.softmax(logits2_0, dim=1), torch.softmax(logits2_1, dim=1),torch.softmax(logits2_2, dim=1)

                    preds1=[]
                    preds1.append(preds1_0)
                    preds1.append(preds1_1)
                    preds1.append(preds1_2)
                    preds2 = []
                    preds2.append(preds2_0)
                    preds2.append(preds2_1)
                    preds2.append(preds2_2)
                    preds1=torch.mean(torch.cat(preds1), 0).unsqueeze(0).cpu().numpy()
                    preds2 = torch.mean(torch.cat(preds2), 0).unsqueeze(0).cpu().numpy()



                    # preds1 = (preds1_0+preds1_1+preds1_2)/3.0  # [1,7,512,512]
                    # preds2 = (preds2_0+preds2_1+preds2_2)/3.0


                    #===i==0=======
                    pred1 = preds1[0]  # [7,512,512]
                    pred2 = preds2[0]
                    if i == 1:
                        # pred1 = preds1[0].copy()[:, ::-1, :]
                        # pred2 = preds2[0].copy()[:, ::-1, :]

                        pred1 = np.rot90(preds1[0].copy(), -1, (1, 2))
                        pred2 = np.rot90(preds2[0].copy(), -1, (1, 2))
                    if i == 2:
                        # pred1 = preds1[0].copy()[:, :, ::-1]
                        # pred2 = preds2[0].copy()[:, :, ::-1]

                        pred1 = np.rot90(preds1[0].copy(), -2, (1, 2))
                        pred2 = np.rot90(preds2[0].copy(), -2, (1, 2))
                    if i == 3:
                        # pred1 = preds1[0].copy()[:, ::-1, ::-1]
                        # pred2 = preds2[0].copy()[:, ::-1, ::-1]

                        pred1 = np.rot90(preds1[0].copy(), -3, (1, 2))
                        pred2 = np.rot90(preds2[0].copy(), -3, (1, 2))
                    preds1_mask.append(pred1)
                    preds2_mask.append(pred2)

                _preds1 = np.average(preds1_mask, axis=0)  # [7,512,512]
                _preds2 = np.average(preds2_mask, axis=0)

                outputs1_label = np.argmax(_preds1, axis=0)
                outputs1_label = outputs1_label.astype('uint8')
                outputs2_label = np.argmax(_preds2, axis=0)
                outputs2_label = outputs2_label.astype('uint8')

                if output_score:
                    infer_list.append(outputs1_label)
                    infer_list.append(outputs2_label)
                    label_list.append(gt1_label)
                    label_list.append(gt2_label)


                if output_rgb:
                    pred1_rgb = np.zeros((outputs1_label.shape + (3,)), dtype='uint8')
                    pred2_rgb = np.zeros((outputs2_label.shape + (3,)), dtype='uint8')

                    for i in range(outputs1_label.shape[0]):
                        for j in range(outputs1_label.shape[1]):

                            key1 = str(outputs1_label[i, j])
                            key2 = str(outputs2_label[i, j])
                            for k in range(3):
                                pred1_rgb[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                                pred2_rgb[i, j, 2 - k] = rgb_table[key2][k]
                    cv2.imwrite(self.config.pred1_rgb_dir + '/' + img_name[0] + '.png', pred1_rgb)
                    cv2.imwrite(self.config.pred2_rgb_dir + '/' + img_name[0] + '.png', pred2_rgb)

                cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', outputs1_label)
                cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', outputs2_label)

        if output_score:
            from utils.SCDD_eval import Eval_preds
            _, _, score = Eval_preds(infer_list, label_list)
            print("score is %.6f" % score)
            return score
        return 0

    def model_out_ensemble2(self, output_rgb=False,
                           output_score=False):  # cannot use for model generated during different training
        import models.Satt_CD.networks as networks
        model_list = []
        dict_list = []

        model0_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc_c0.pth'
        model1_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc_c1.pth'
        model2_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc_c2.pth'

        model0 = networks.define_G_CD(self.config).to(self.device)  # must define model instead of using self.net
        model1 = networks.define_G_CD(self.config).to(self.device)
        model2 = networks.define_G_CD(self.config).to(self.device)

        model0.load_state_dict(torch.load(model0_path))
        model1.load_state_dict(torch.load(model1_path))
        model2.load_state_dict(torch.load(model2_path))

        model_list.append(model0)
        model_list.append(model1)
        model_list.append(model2)

        rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                     '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}

        infer_list = []
        label_list = []

        preds1_model = []
        pred2_model = []
        for model in model_list:
            for i, sample in enumerate(tqdm(self.testDataloader, 0)):
                with torch.no_grad():
                    img_name = sample['name']
                    imgs, labels = sample['img'], sample['label']
                    imgs, labels = imgs.cuda(), labels.cuda()
                    print("processing image {}, size is {}".format(i, imgs.shape))
                    if output_score:
                        gt1_label = labels[0, 0, :, :].data.cpu().numpy().astype('uint8')  # [1,512,512]
                        gt2_label = labels[0, 1, :, :].data.cpu().numpy().astype('uint8')

                    imgs_aug = []
                    imgs = imgs.data.cpu().numpy()
                    imgs_aug.append(imgs)
                    imgs_aug.append(np.rot90(imgs, 1, (2, 3)))
                    imgs_aug.append(np.rot90(imgs, 2, (2, 3)))
                    imgs_aug.append(np.rot90(imgs, 3, (2, 3)))
                    preds1_mask = []
                    preds2_mask = []


                    for i in range(len(imgs_aug)):
                        cur_img = torch.from_numpy(imgs_aug[i].copy()).cuda()  # must use .copy() after flip operation
                        if self.config["train"]["use_label_rgb255"]:
                            _, _, _, (logits1, logits2, _) = model(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        elif self.config["train"]["use_CatOut"]:
                            _, _, _, _, (logits1, logits2) = model(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        else:
                            _, _, _, (logits1, logits2) = model(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])

                            preds1, preds2 = torch.softmax(logits1, dim=1).cpu().numpy(), torch.softmax(logits2,dim=1).cpu().numpy()

                            # ===i==0=======
                            pred1 = preds1[0]  # [7,512,512]
                            pred2 = preds2[0]
                            if i == 1:
                                # pred1 = preds1[0].copy()[:, ::-1, :]
                                # pred2 = preds2[0].copy()[:, ::-1, :]

                                pred1 = np.rot90(preds1[0].copy(), -1, (1, 2))
                                pred2 = np.rot90(preds2[0].copy(), -1, (1, 2))
                            if i == 2:
                                # pred1 = preds1[0].copy()[:, :, ::-1]
                                # pred2 = preds2[0].copy()[:, :, ::-1]

                                pred1 = np.rot90(preds1[0].copy(), -2, (1, 2))
                                pred2 = np.rot90(preds2[0].copy(), -2, (1, 2))
                            if i == 3:
                                # pred1 = preds1[0].copy()[:, ::-1, ::-1]
                                # pred2 = preds2[0].copy()[:, ::-1, ::-1]

                                pred1 = np.rot90(preds1[0].copy(), -3, (1, 2))
                                pred2 = np.rot90(preds2[0].copy(), -3, (1, 2))
                            preds1_mask.append(pred1)
                            preds2_mask.append(pred2)

                        _preds1 = np.average(preds1_mask, axis=0)  # [7,512,512]
                        _preds2 = np.average(preds2_mask, axis=0)




                outputs1_label = np.argmax(_preds1, axis=0)
                outputs1_label = outputs1_label.astype('uint8')
                outputs2_label = np.argmax(_preds2, axis=0)
                outputs2_label = outputs2_label.astype('uint8')

                if output_score:
                    infer_list.append(outputs1_label)
                    infer_list.append(outputs2_label)
                    label_list.append(gt1_label)
                    label_list.append(gt2_label)

                if output_rgb:
                    pred1_rgb = np.zeros((outputs1_label.shape + (3,)), dtype='uint8')
                    pred2_rgb = np.zeros((outputs2_label.shape + (3,)), dtype='uint8')

                    for i in range(outputs1_label.shape[0]):
                        for j in range(outputs1_label.shape[1]):

                            key1 = str(outputs1_label[i, j])
                            key2 = str(outputs2_label[i, j])
                            for k in range(3):
                                pred1_rgb[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                                pred2_rgb[i, j, 2 - k] = rgb_table[key2][k]
                    cv2.imwrite(self.config.pred1_rgb_dir + '/' + img_name[0] + '.png', pred1_rgb)
                    cv2.imwrite(self.config.pred2_rgb_dir + '/' + img_name[0] + '.png', pred2_rgb)

                cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', outputs1_label)
                cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', outputs2_label)

        if output_score:
            from utils.SCDD_eval import Eval_preds
            _, _, score = Eval_preds(infer_list, label_list)
            print("score is %.6f" % score)
            return score
        return 0



    def inferSenCD2_label7_TTA(self,use_model_ensemble=False,output_rgb=False):
            '''
            output y_true,y_pred, y_pred_p(of probability not 0,1)
            :return:
            '''

            self.net.eval()

            if use_model_ensemble:
               self.model_para_ensemble()
            else:
               model_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc.pth'
               self.net.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络

            rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                         '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}

            '''
            This means that your numpy array has undergone such operation:
    image = image[..., ::-1]
    I guess this has something to do with how numpy array are stored in memory, and unfortunately PyTorch doesn’t currently support numpy array that has been reversed using negative stride.

    A simple fix is to do

    image = image[..., ::-1] - np.zeros_like(image)
    https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/10
            '''
            infer_list = []
            label_list = []

            for i, sample in enumerate(tqdm(self.testDataloader, 0)):
                with torch.no_grad():
                    img_name = sample['name']
                    imgs= sample['img']
                    imgs= imgs.cuda()
                    print("processing image {}, size is {}".format(i, imgs.shape))
                    # gt1_label = labels[0, 0, :, :].data.cpu().numpy().astype('uint8')  # [1,512,512]
                    # gt2_label = labels[0, 1, :, :].data.cpu().numpy().astype('uint8')

                    imgs_aug = []
                    imgs = imgs.data.cpu().numpy()
                    imgs_aug.append(imgs)
                    imgs_aug.append(imgs[:, :, ::-1, :])
                    imgs_aug.append(imgs[:, :, :, ::-1])
                    imgs_aug.append(imgs[:, :, ::-1, ::-1])
                    preds1_mask = []
                    preds2_mask = []

                    for i in range(4):
                        cur_img = torch.from_numpy(imgs_aug[i].copy()).cuda()  # must use .copy() after flip operation
                        if self.config["train"]["use_label_rgb255"]:
                            _, _, _, (logits1, logits2, _) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        elif self.config["train"]["use_CatOut"] or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5':
                            _, _, _,_ ,(logits1, logits2) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        else:
                            _, _, _, (logits1, logits2) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])

                        preds1 = torch.softmax(logits1, dim=1).cpu().numpy()  # [1,7,512,512]
                        preds2 = torch.softmax(logits2, dim=1).cpu().numpy()
                        pred1 = preds1[0]  # [7,512,512]
                        pred2 = preds2[0]
                        if i == 1:
                            pred1 = preds1[0].copy()[:, ::-1, :]
                            pred2 = preds2[0].copy()[:, ::-1, :]
                        if i == 2:
                            pred1 = preds1[0].copy()[:, :, ::-1]
                            pred2 = preds2[0].copy()[:, :, ::-1]
                        if i == 3:
                            pred1 = preds1[0].copy()[:, ::-1, ::-1]
                            pred2 = preds2[0].copy()[:, ::-1, ::-1]
                        preds1_mask.append(pred1)
                        preds2_mask.append(pred2)

                    _preds1 = np.average(preds1_mask, axis=0)  # [7,512,512]
                    _preds2 = np.average(preds2_mask, axis=0)

                    outputs1_label = np.argmax(_preds1, axis=0)
                    outputs1_label = outputs1_label.astype('uint8')
                    outputs2_label = np.argmax(_preds2, axis=0)
                    outputs2_label = outputs2_label.astype('uint8')

                    if output_rgb:
                        pred1_rgb = np.zeros((outputs1_label.shape + (3,)), dtype='uint8')
                        pred2_rgb = np.zeros((outputs2_label.shape + (3,)), dtype='uint8')

                        for i in range(outputs1_label.shape[0]):
                            for j in range(outputs1_label.shape[1]):

                                key1 = str(outputs1_label[i, j])
                                key2 = str(outputs2_label[i, j])
                                for k in range(3):
                                    pred1_rgb[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                                    pred2_rgb[i, j, 2 - k] = rgb_table[key2][k]
                        cv2.imwrite(self.config.pred1_rgb_dir + '/' + img_name[0] + '.png', pred1_rgb)
                        cv2.imwrite(self.config.pred2_rgb_dir + '/' + img_name[0] + '.png', pred2_rgb)

                    cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', outputs1_label)
                    cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', outputs2_label)



    def inferSenCD2_label7_TTA_RGB(self,use_model_ensemble=False):
            '''
            output y_true,y_pred, y_pred_p(of probability not 0,1)
            :return:
            '''

            self.net.eval()

            if use_model_ensemble:
               self.model_para_ensemble()
            else:
               model_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc.pth'
               self.net.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络

            rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                         '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}

            for i, sample in enumerate(tqdm(self.testDataloader, 0)):
                with torch.no_grad():
                    img_name = sample['name']
                    imgs= sample['img']
                    imgs= imgs.cuda()
                    print("processing image {}, size is {}".format(i, imgs.shape))


                    imgs_aug = []
                    imgs = imgs.data.cpu().numpy()
                    imgs_aug.append(imgs)
                    imgs_aug.append(imgs[:, :, ::-1, :])
                    imgs_aug.append(imgs[:, :, :, ::-1])
                    imgs_aug.append(imgs[:, :, ::-1, ::-1])

                    preds1_mask = []
                    preds2_mask = []
                    for i in range(4):
                        cur_img = torch.from_numpy(imgs_aug[i].copy()).cuda()  # must use .copy() after flip operation
                        if self.config["train"]["use_label_rgb255"]:
                            _, _, _, (logits1, logits2, _) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        else:
                            _, _, _, (logits1, logits2) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])

                        preds1 = torch.softmax(logits1, dim=1).cpu().numpy()  # [1,7,512,512]
                        preds2 = torch.softmax(logits2, dim=1).cpu().numpy()
                        pred1 = preds1[0]  # [7,512,512]
                        pred2 = preds2[0]
                        if i == 1:
                            pred1 = preds1[0].copy()[:, ::-1, :]
                            pred2 = preds2[0].copy()[:, ::-1, :]
                        if i == 2:
                            pred1 = preds1[0].copy()[:, :, ::-1]
                            pred2 = preds2[0].copy()[:, :, ::-1]
                        if i == 3:
                            pred1 = preds1[0].copy()[:, ::-1, ::-1]
                            pred2 = preds2[0].copy()[:, ::-1, ::-1]
                        preds1_mask.append(pred1)
                        preds2_mask.append(pred2)

                    _preds1 = np.average(preds1_mask, axis=0)  # [7,512,512]
                    _preds2 = np.average(preds2_mask, axis=0)

                    outputs1_label = np.argmax(_preds1, axis=0)
                    outputs1_label = outputs1_label.astype('uint8')
                    outputs2_label = np.argmax(_preds2, axis=0)
                    outputs2_label = outputs2_label.astype('uint8')

                    pred1_rgb = np.zeros((outputs1_label.shape + (3,)), dtype='uint8')
                    pred2_rgb = np.zeros((outputs2_label.shape + (3,)), dtype='uint8')

                    for i in range(outputs1_label.shape[0]):
                        for j in range(outputs1_label.shape[1]):

                            key1 = str(outputs1_label[i, j])
                            key2 = str(outputs2_label[i, j])
                            for k in range(3):
                                pred1_rgb[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                                pred2_rgb[i, j, 2 - k] = rgb_table[key2][k]

                    cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', pred1_rgb)
                    cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', pred2_rgb)





    def inferSenCD2_score_label7_TTA(self,use_model_ensemble=False,output_rgb=False,mode='_best_acc'):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        it seems that when use randomscale for training, rot90TTA is better than flipTTA, in fact rot90TTA behaves better in all cases

        '''

        self.net.eval()
        if use_model_ensemble:
            self.model_para_ensemble()

            # score=self.model_out_ensemble(output_rgb=output_rgb,output_score=True)
            # return score
        else:
            mode_snap=mode
            model_path = self.config.model_dir + '/' + self.config.pred_name +mode_snap+ '.pth'
            self.net.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络
        rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                     '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}


        '''
        This means that your numpy array has undergone such operation:
image = image[..., ::-1]
I guess this has something to do with how numpy array are stored in memory, and unfortunately PyTorch doesn’t currently support numpy array that has been reversed using negative stride.

A simple fix is to do

image = image[..., ::-1] - np.zeros_like(image)
https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/10
        '''
        infer_list=[]
        label_list=[]

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):
            with torch.no_grad():
                img_name = sample['name']
                imgs, labels = sample['img'], sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                print("processing image {}, size is {}".format(i, imgs.shape))
                gt1_label = labels[0, 0, :, :].data.cpu().numpy().astype('uint8')#[1,512,512]
                gt2_label = labels[0, 1, :, :].data.cpu().numpy().astype('uint8')

                imgs_aug=[]
                imgs=imgs.data.cpu().numpy()
                imgs_aug.append(imgs)
                #imgs_aug.append(imgs[:,:,::-1,:])
                # imgs_aug.append(imgs[:, :, :, ::-1])
                # imgs_aug.append(imgs[:, :, ::-1, ::-1])

                imgs_aug.append(np.rot90(imgs,1,(2,3)))
                imgs_aug.append(np.rot90(imgs, 2,(2,3)))
                imgs_aug.append(np.rot90(imgs, 3,(2,3)))

                preds1_mask=[]
                preds2_mask=[]
                for i in range(len(imgs_aug)):
                    cur_img=torch.from_numpy(imgs_aug[i].copy()).cuda()#must use .copy() after flip operation
                    if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3':
                        _, _, _, (logits1, logits2,_) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                    elif self.config["train"]["use_CatOut"] or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5':
                        _, _, _,_ ,(logits1, logits2) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                    else:
                        _, _, _, (logits1, logits2) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])

                    preds1=torch.softmax(logits1,dim=1).cpu().numpy()#[1,7,512,512]
                    preds2 = torch.softmax(logits2, dim=1).cpu().numpy()
                    pred1 = preds1[0]#[7,512,512]
                    pred2=preds2[0]
                    if i == 1:
                        # pred1 = preds1[0].copy()[:, ::-1, :]
                        # pred2 = preds2[0].copy()[:, ::-1, :]

                        pred1 = np.rot90(preds1[0].copy(),-1,(1,2))
                        pred2 = np.rot90(preds2[0].copy(),-1,(1,2))
                    if i == 2:
                        # pred1 = preds1[0].copy()[:, :, ::-1]
                        # pred2 = preds2[0].copy()[:, :, ::-1]

                        pred1 = np.rot90(preds1[0].copy(), -2,(1,2))
                        pred2 = np.rot90(preds2[0].copy(), -2,(1,2))
                    if i == 3:
                        # pred1 = preds1[0].copy()[:, ::-1, ::-1]
                        # pred2 = preds2[0].copy()[:, ::-1, ::-1]

                        pred1 = np.rot90(preds1[0].copy(), -3,(1,2))
                        pred2 = np.rot90(preds2[0].copy(), -3,(1,2))
                    preds1_mask.append(pred1)
                    preds2_mask.append(pred2)

                _preds1 = np.average(preds1_mask, axis=0)#[7,512,512]
                _preds2 = np.average(preds2_mask, axis=0)

                outputs1_label=np.argmax(_preds1,axis=0)
                outputs1_label=outputs1_label.astype('uint8')
                outputs2_label = np.argmax(_preds2, axis=0)
                outputs2_label = outputs2_label.astype('uint8')


                infer_list.append(outputs1_label)
                infer_list.append(outputs2_label)
                label_list.append(gt1_label)
                label_list.append(gt2_label)

                if output_rgb:
                    pred1_rgb = np.zeros((outputs1_label.shape + (3,)), dtype='uint8')
                    pred2_rgb = np.zeros((outputs2_label.shape + (3,)), dtype='uint8')

                    for i in range(outputs1_label.shape[0]):
                        for j in range(outputs1_label.shape[1]):

                            key1 = str(outputs1_label[i, j])
                            key2 = str(outputs2_label[i, j])
                            for k in range(3):
                                pred1_rgb[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                                pred2_rgb[i, j, 2 - k] = rgb_table[key2][k]
                    cv2.imwrite(self.config.pred1_rgb_dir + '/' + img_name[0] + '.png', pred1_rgb)
                    cv2.imwrite(self.config.pred2_rgb_dir + '/' + img_name[0] + '.png', pred2_rgb)

                

                cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', outputs1_label)
                cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', outputs2_label)


        from utils.SCDD_eval import Eval_preds
        _, _, score = Eval_preds(infer_list, label_list)
        print("score is %.6f" % score)
        return  score

    def load_ck(self, model, model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        loaded_dict = checkpoint
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
    def inferSenCD2_score_label7_TTA_BT(self, use_TTA=True,use_model_ensemble=False, output_rgb=False, use_con=False,use_score=False,mode='_best_acc',use_CRF=False):
            '''
            testing using batchsize>1
            :return:
            it seems that when use randomscale for training, rot90TTA is better than flipTTA, in fact rot90TTA behaves better in all cases
    #==========================for france data===========================
    - 0: No information ==>(255,255,255)

    - 1: Artificial surfaces==> Firebrick	178 34 34

    - 2: Agricultural areas==> (238, 238, 0) Yellow2

    - 3: Forests==>(34 139 34)  ForestGreen

    - 4: Wetlands==> (0,139,139) DarkCyan

    - 5: Water==>（0,0,255）
    #=============for sensetime data=======================================
    未变化区域（255,255,255）0
水体（0,0,255）1
地面（128,128,128）2
低矮植被（0,128,0）3
树木（0,255,0）4
建筑物（128,0,0）5
运动场（255,0,0）6
            '''
            #from utils.utils import area_connection

            self.net.eval()
            if use_model_ensemble:
                self.model_para_ensemble()
            else:
                mode_snap = mode
                model_path = self.config.model_dir + '/' + self.config.pred_name + mode_snap + '.pth'
                self.net.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络
            # change_type = ['0_0',
            #                '1_2', '1_3', '1_4', '1_5', '1_6',
            #                '2_1', '2_3', '2_4', '2_5', '2_6',
            #                '3_1', '3_2', '3_4', '3_5', '3_6',
            #                '4_1', '4_2', '4_3', '4_5', '4_6',
            #                '5_1', '5_2', '5_3', '5_4', '5_5', '5_6',
            #                '6_1', '6_2', '6_3', '6_4', '6_5']

            if self.config["dataset_name"] == 'sensetime':
                rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                             '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}  # for sensetime
            else:
                # rgb_table = {'0': (255, 255, 255), '1': (178, 34, 34), '2': (238, 238, 0), '3': (34, 139, 34),
                #          '4': (107, 142, 35), '5': (0, 0, 255)}  # for france
                rgb_table = {'0': (255, 255, 255), '1': (178, 34, 34), '2': (238, 238, 0), '3': (34, 139, 34),
                             '4': (0, 139, 139), '5': (0, 0, 255)}





            infer_list = []
            label_list = []

            if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':
                import models.Satt_CD.networks as networks
                netCD = networks.define_G_CD0(self.config).to(self.device)
                self.load_ck(netCD, self.config.pretrained_model_path)
                netCD.eval()

            for i, sample in enumerate(tqdm(self.testDataloader, 0)):
                with torch.no_grad():
                    img_name = sample['name']
                    if use_score:
                        imgs, labels = sample['img'], sample['label']
                        imgs, labels = imgs.cuda(), labels.cuda()
                        #print("processing image {}, size is {}".format(i, imgs.shape))
                        gt1_label = labels[0, 0, :, :].data.cpu().numpy().astype('uint8')  # [1,512,512]
                        gt2_label = labels[0, 1, :, :].data.cpu().numpy().astype('uint8')
                    else:
                        imgs = sample['img']
                        imgs = imgs.cuda()


                    if use_TTA:
                        imgs_aug = []
                        preds1_mask = []
                        preds2_mask = []
                        preds12_mask=[]
                        # ======pred imgs_aug simultaneously instead of one-by-one==============================


                        if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':

                            images_T1, images_T2 = imgs[:, 0:3, ...], imgs[:, 3:6, ...]
                            with torch.no_grad():
                                _, _, _, images_cd = netCD(images_T1, images_T2)

                            images_T1 = torch.cat([images_T1, images_cd], dim=1)
                            images_T2 = torch.cat([images_T2, images_cd], dim=1)
                            imgs_new=torch.cat([images_T1,images_T2],dim=1)
                            imgs_new = imgs_new.data.cpu().numpy()
                            imgs_aug = [imgs_new
                                        ,np.rot90(imgs_new, 1, (2, 3)), np.rot90(imgs_new, 2, (2, 3)), np.rot90(imgs_new, 3, (2, 3))
                                        ]
                            cur_img = torch.from_numpy(np.array(imgs_aug).copy()).squeeze(
                                1).cuda()  # must use .copy() after flip operation
                            logits1, logits2 = self.net(cur_img[:, 0:4, ...], cur_img[:, 4:8, ...])


                        elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin'\
                                or self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                            imgs = imgs.data.cpu().numpy()
                            imgs_aug = [imgs,
                                        np.rot90(imgs, 1, (2, 3)), np.rot90(imgs, 2, (2, 3)), np.rot90(imgs, 3, (2, 3))
                                        # imgs[:, :, ::-1, :],imgs[:, :, :, ::-1],imgs[:, :, ::-1, ::-1]
                                        ]
                            cur_img = torch.from_numpy(np.array(imgs_aug).copy()).squeeze(
                                1).cuda()  # must use .copy() after flip operation

                            logits1,logits2,logits12 = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])#test mode, not, _,_,_,()


                        else:
                            imgs = imgs.data.cpu().numpy()
                            imgs_aug = [imgs,
                                        np.rot90(imgs, 1, (2, 3)), np.rot90(imgs, 2, (2, 3)), np.rot90(imgs, 3, (2, 3))
                                        # imgs[:, :, ::-1, :],imgs[:, :, :, ::-1],imgs[:, :, ::-1, ::-1]
                                        ]
                            cur_img = torch.from_numpy(np.array(imgs_aug).copy()).squeeze(
                                1).cuda()  # must use .copy() after flip operation

                            logits1, logits2 = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])  # for test mode

                        if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin'\
                                or self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD' :
                            preds1 = torch.softmax(logits1, dim=1).cpu().numpy()  # [4,7,512,512]
                            preds2 = torch.softmax(logits2, dim=1).cpu().numpy()
                            preds12=logits12.cpu().numpy()
                            for i in range(len(imgs_aug)):
                                if i == 0:
                                    pred1 = preds1[i]
                                    pred2 = preds2[i]
                                    pred12=preds12[i]
                                if i == 1:
                                    pred1 = np.rot90(preds1[i].copy(), -1, (1, 2))
                                    pred2 = np.rot90(preds2[i].copy(), -1, (1, 2))
                                    pred12=np.rot90(preds12[i].copy(), -1, (1, 2))
                                if i == 2:
                                    pred1 = np.rot90(preds1[i].copy(), -2, (1, 2))
                                    pred2 = np.rot90(preds2[i].copy(), -2, (1, 2))
                                    pred12 = np.rot90(preds12[i].copy(), -2, (1, 2))
                                if i == 3:
                                    pred1 = np.rot90(preds1[i].copy(), -3, (1, 2))
                                    pred2 = np.rot90(preds2[i].copy(), -3, (1, 2))
                                    pred12 = np.rot90(preds12[i].copy(), -3, (1, 2))

                                preds1_mask.append(pred1)
                                preds2_mask.append(pred2)
                                preds12_mask.append(pred12)

                            _preds1 = np.average(preds1_mask, axis=0)  # [7,512,512]
                            _preds2 = np.average(preds2_mask, axis=0)
                            _preds12 = np.average(preds12_mask, axis=0)[0]
                        else:
                            preds1 = torch.softmax(logits1, dim=1).cpu().numpy()  # [4,7,512,512]
                            preds2 = torch.softmax(logits2, dim=1).cpu().numpy()
                            for i in range(len(imgs_aug)):
                                if i == 0:
                                    pred1 = preds1[i]
                                    pred2 = preds2[i]
                                if i == 1:
                                    pred1 = np.rot90(preds1[i].copy(), -1, (1, 2))
                                    pred2 = np.rot90(preds2[i].copy(), -1, (1, 2))
                                if i == 2:
                                    pred1 = np.rot90(preds1[i].copy(), -2, (1, 2))
                                    pred2 = np.rot90(preds2[i].copy(), -2, (1, 2))
                                if i == 3:
                                    pred1 = np.rot90(preds1[i].copy(), -3, (1, 2))
                                    pred2 = np.rot90(preds2[i].copy(), -3, (1, 2))
                                # if i == 4:
                                #     pred1 = preds1[i].copy()[:, ::-1, :]
                                #     pred2 = preds2[i].copy()[:, ::-1, :]
                                # if i == 5:
                                #     pred1 = preds1[i].copy()[:, :, ::-1]
                                #     pred2 = preds2[i].copy()[:, :, ::-1]
                                # if i == 6:
                                #     pred1 = preds1[i].copy()[:, ::-1, ::-1]
                                #     pred2 = preds2[i].copy()[:, ::-1, ::-1]

                                preds1_mask.append(pred1)
                                preds2_mask.append(pred2)

                            _preds1 = np.average(preds1_mask, axis=0)  # [7,512,512]
                            _preds2 = np.average(preds2_mask, axis=0)




                    else:
                        logits1, logits2 = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])  # for test mode
                        _preds1 = torch.softmax(logits1, dim=1).cpu().numpy()[0]  # [7,512,512]
                        _preds2 = torch.softmax(logits2, dim=1).cpu().numpy()[0]


                    if use_CRF:
                        _preds1=self.infer_CRF(imgs[0,0:3,...],_preds1)
                        _preds2 = self.infer_CRF(imgs[0,3:6,...], _preds2)


                    outputs1_label = np.argmax(_preds1, axis=0)
                    outputs1_label = outputs1_label.astype('uint8')
                    outputs2_label = np.argmax(_preds2, axis=0)
                    outputs2_label = outputs2_label.astype('uint8')
                    if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin' or\
                        self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                        if self.config["train"]["use_MC6"]:
                            outputs1_label+=1
                            outputs2_label+=1
                            outputs1_label[_preds12 < 0.5] = 0
                            outputs2_label[_preds12 < 0.5] = 0
                        else:
                            outputs1_label[_preds12 < 0.5] = 0
                            outputs2_label[_preds12< 0.5] = 0
                    # if use_con:
                    #     mask0_1=(outputs1_label==0)
                    #     mask0_2=(outputs2_label==0)
                    #     #outputs2_label_new=outputs2_label.copy()
                    #     outputs1_label[mask0_2] = 0
                    #     outputs2_label[mask0_1]=0



                    if use_score:
                        infer_list.append(outputs1_label)
                        infer_list.append(outputs2_label)
                        label_list.append(gt1_label)
                        label_list.append(gt2_label)

                    if output_rgb:
                        pred1_rgb = np.zeros((outputs1_label.shape + (3,)), dtype='uint8')
                        pred2_rgb = np.zeros((outputs2_label.shape + (3,)), dtype='uint8')

                        for i in range(outputs1_label.shape[0]):
                            for j in range(outputs1_label.shape[1]):

                                key1 = str(outputs1_label[i, j])
                                key2 = str(outputs2_label[i, j])
                                for k in range(3):
                                    pred1_rgb[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                                    pred2_rgb[i, j, 2 - k] = rgb_table[key2][k]
                        cv2.imwrite(self.config.pred1_rgb_dir + '/' + img_name[0] + '.png', pred1_rgb)
                        cv2.imwrite(self.config.pred2_rgb_dir + '/' + img_name[0] + '.png', pred2_rgb)

                    cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', outputs1_label)
                    cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', outputs2_label)
            if use_score:
                from utils.SCDD_eval import Eval_preds
                IoU_mean, Sek, Score = Eval_preds(infer_list, label_list)
                print("mIOU is %.6f" % IoU_mean)
                print("Sek is %.6f" % Sek)
                print("score is %.6f" % Score)
                return IoU_mean, Sek, Score


    def infer_CRF(self,image,unary,use_norm=False):
        # get basic hyperparameters

        num_classes = unary.shape[0]
        shape = image.shape[1:]
        config = convcrf.default_conf
        config['filter_size'] = 7#for sensetime data, 7 works bettern than 5
        if use_norm:
            #    CRF is invariant to mean subtraction, output is NOT affected
            image = image - 0.5
            # std normalization
            #       Affect CRF computation
            image = image / 0.3

            # schan = 0.1 is a good starting value for normalized images.
            # The relation is f_i = image * schan
            config['col_feats']['schan'] = 0.1
        else:
            image = image * 255  # not use normalization



        gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes,
                                    use_gpu=True)
        # Perform CRF inference
        image = image.reshape([1, 3, shape[0], shape[1]])
        img_var = Variable(torch.Tensor(image))

        # Add batch dimension to unary: [1, 21, height, width]
        unary = unary.reshape([1, num_classes, shape[0], shape[1]])
        unary_var = Variable(torch.Tensor(unary))
        #cuda
        img_var = img_var.cuda()
        unary_var = unary_var.cuda()
        gausscrf.cuda()
        prediction = gausscrf.forward(unary=unary_var, img=img_var)

        return prediction[0].data.cpu().numpy()










    def inferSenCD2_score_label7_TTA_BT_Com(self, use_model_ensemble=False, output_rgb=False, use_score=False,use_filter=False,mode='_best_acc'):
            '''
            testing using batchsize>1
            :return:
            it seems that when use randomscale for training, rot90TTA is better than flipTTA, in fact rot90TTA behaves better in all cases
            '''

            self.net.eval()
            if use_model_ensemble:
                #====================for model parameters ensemble=====================
                #self.model_para_ensemble()

                #===============for model output ensemble from different models==========================
                print("using model output ensemble...")
                # import models.Satt_CD.networks as networks
                # models=[]
                # models.append(self.net)
                # #model0_name='netG_Res34_aug5_mixscale_SE_EDCls_UNet2_New_diffmode_diff_attmode_BAM_dtype_AS_Drop_0.10_patch_512_batch_4_nepoch_30_warmepoch_4_scheduler_CosineLR'
                # #model0_name='netG_Res34_aug5_mixscale_EDCls_UNet2_diffmode_diff_attmode_BAM_actmode_relu_Drop_0.10_patch_512_batch_4_nepoch_25_warmepoch_4_scheduler_CosineLR'
                # #model0_path=self.config.model_dir + '/' +model0_name + '_best_acc.pth'
                # model0_path = self.config.model_dir + '/' + self.config.pred_name + '_final_iter.pth'
                # model0 = networks.define_G_CD(self.config).to(self.device)
                # model0.load_state_dict(torch.load(model0_path))
                # model0.eval()
                # models.append(model0)
                model_path=[]
                model_name0='netG_Res34_mixscale_EDCls_UNet2_New2_diffmode_diff_dtype_AS_se_BAM_Drop_0.10_ce_weight_4.00_patch_512_batch_4_nepoch_5_warmepoch_0'
                model_name1='netG_Res34_aug5_mixscale_EDCls_UNet2_diffmode_diff_attmode_BAM_actmode_relu_Drop_0.10_patch_512_batch_4_nepoch_25_warmepoch_4_scheduler_CosineLR'
                model_path.append(self.config.model_dir + '/' +model_name0 + '_best_acc.pth')
                model_path.append(self.config.model_dir + '/' +model_name0 + '_final_iter.pth')


            else:
                mode_snap = mode
                model_path = self.config.model_dir + '/' + self.config.pred_name + mode_snap + '.pth'
                self.net.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络
            rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                         '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}

            infer_list = []
            label_list = []

            for i, sample in enumerate(tqdm(self.testDataloader, 0)):
                with torch.no_grad():
                    img_name = sample['name']
                    if use_score:
                        imgs, labels = sample['img'], sample['label']
                        imgs, labels = imgs.cuda(), labels.cuda()
                        print("processing image {}, size is {}".format(i, imgs.shape))
                        gt1_label = labels[0, 0, :, :].data.cpu().numpy().astype('uint8')  # [1,512,512]
                        gt2_label = labels[0, 1, :, :].data.cpu().numpy().astype('uint8')
                    else:
                        imgs = sample['img']
                        imgs = imgs.cuda()

                    imgs_aug = []
                    imgs = imgs.data.cpu().numpy()
                    imgs_aug=[imgs,
                              np.rot90(imgs, 1, (2, 3)),np.rot90(imgs, 2, (2, 3)),np.rot90(imgs, 3, (2, 3))
                              #imgs[:, :, ::-1, :],imgs[:, :, :, ::-1],imgs[:, :, ::-1, ::-1]
                              ]

                    preds1_mask = []
                    preds2_mask = []

                    #======pred imgs_aug simultaneously instead of one-by-one==============================
                    cur_img = torch.from_numpy(np.array(imgs_aug).copy()).squeeze(1).cuda()  # must use .copy() after flip operation
                    for path in model_path:
                        self.net.load_state_dict(torch.load(path))
                        logits1, logits2 = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        #_, _, _, (logits1, logits2) = self.net(cur_img[:, 0:3, ...], cur_img[:, 3:6, ...])
                        preds1 = torch.softmax(logits1, dim=1).cpu().numpy()  # [4,7,512,512]
                        preds2 = torch.softmax(logits2, dim=1).cpu().numpy()

                        for i in range(len(imgs_aug)):
                            if i == 0:
                                pred1 = preds1[i]
                                pred2 = preds2[i]
                            if i == 1:
                                pred1 = np.rot90(preds1[i].copy(), -1, (1, 2))
                                pred2 = np.rot90(preds2[i].copy(), -1, (1, 2))
                            if i == 2:
                                pred1 = np.rot90(preds1[i].copy(), -2, (1, 2))
                                pred2 = np.rot90(preds2[i].copy(), -2, (1, 2))
                            if i == 3:
                                pred1 = np.rot90(preds1[i].copy(), -3, (1, 2))
                                pred2 = np.rot90(preds2[i].copy(), -3, (1, 2))
                            # if i == 4:
                            #     pred1 = preds1[i].copy()[:, ::-1, :]
                            #     pred2 = preds2[i].copy()[:, ::-1, :]
                            # if i == 5:
                            #     pred1 = preds1[i].copy()[:, :, ::-1]
                            #     pred2 = preds2[i].copy()[:, :, ::-1]
                            # if i == 6:
                            #     pred1 = preds1[i].copy()[:, ::-1, ::-1]
                            #     pred2 = preds2[i].copy()[:, ::-1, ::-1]

                            preds1_mask.append(pred1)
                            preds2_mask.append(pred2)

                    _preds1 = np.average(preds1_mask, axis=0)  # [7,512,512]
                    _preds2 = np.average(preds2_mask, axis=0)

                    outputs1_label = np.argmax(_preds1, axis=0)
                    outputs1_label = outputs1_label.astype('uint8')
                    outputs2_label = np.argmax(_preds2, axis=0)
                    outputs2_label = outputs2_label.astype('uint8')

                    # if use_filter:
                    #     outputs1_label=cv2.medianBlur(outputs1_label,21)
                    #     outputs2_label = cv2.medianBlur(outputs2_label, 21)


                    if use_score:
                        infer_list.append(outputs1_label)
                        infer_list.append(outputs2_label)
                        label_list.append(gt1_label)
                        label_list.append(gt2_label)

                    if output_rgb:
                        pred1_rgb = np.zeros((outputs1_label.shape + (3,)), dtype='uint8')
                        pred2_rgb = np.zeros((outputs2_label.shape + (3,)), dtype='uint8')

                        for i in range(outputs1_label.shape[0]):
                            for j in range(outputs1_label.shape[1]):

                                key1 = str(outputs1_label[i, j])
                                key2 = str(outputs2_label[i, j])
                                for k in range(3):
                                    pred1_rgb[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                                    pred2_rgb[i, j, 2 - k] = rgb_table[key2][k]
                        cv2.imwrite(self.config.pred1_rgb_dir + '/' + img_name[0] + '.png', pred1_rgb)
                        cv2.imwrite(self.config.pred2_rgb_dir + '/' + img_name[0] + '.png', pred2_rgb)

                    cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', outputs1_label)
                    cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', outputs2_label)
            if use_score:
                from utils.SCDD_eval import Eval_preds
                IoU_mean, Sek, Score = Eval_preds(infer_list, label_list)
                print("mIOU is %.6f" % IoU_mean)
                print("Sek is %.6f" % Sek)
                print("score is %.6f" % Score)
                return IoU_mean, Sek, Score



    def inferSenCD2_score_label7(self):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        '''

        self.net.eval()
        model_path = self.config.model_dir + '/' + self.config.pred_name + '_final_iter.pth'
        self.net.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络

        # change_type = ['0_0',
        #                '1_2', '1_3', '1_4', '1_5', '1_6',
        #                '2_1', '2_3', '2_4', '2_5', '2_6',
        #                '3_1', '3_2', '3_4', '3_5', '3_6',
        #                '4_1', '4_2', '4_3', '4_5', '4_6',
        #                '5_1', '5_2', '5_3', '5_4', '5_5', '5_6',
        #                '6_1', '6_2', '6_3', '6_4', '6_5']
        # rgb_table={'0':(255,255,255),'1':(0,0,255),'2':(128,128,128),'3':(0,128,0),
        #            '4':(0,255,0),'5':(128,0,0),'6':(255,0,0)}

        infer_list = []
        label_list = []

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):
            with torch.no_grad():
                img_name = sample['name']
                imgs, labels = sample['img'], sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                print("processing image {}, size is {}".format(i, imgs.shape))
                gt1_label = labels[:, 0, :, :].data.cpu().numpy().astype('uint8')
                gt2_label = labels[:, 1, :, :].data.cpu().numpy().astype('uint8')

                if self.config["train"]["use_DS"]:
                    _, _, _, (outputs1, outputs2) = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                else:
                    outputs1, outputs2 = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                # outputs1, outputs2 = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                outputs1_label = torch.argmax(outputs1, dim=1)  # [4,32,256,256]==>[4,256,256]
                outputs1_label = outputs1_label.data.cpu().numpy().astype('uint8')

                outputs2_label = torch.argmax(outputs2, dim=1)
                outputs2_label = outputs2_label.data.cpu().numpy().astype('uint8')

                infer_list.append(outputs1_label)
                infer_list.append(outputs2_label)
                label_list.append(gt1_label)
                label_list.append(gt2_label)

                cv2.imwrite(self.config.pred1_dir + '/' + img_name[0] + '.png', outputs1_label[0])
                cv2.imwrite(self.config.pred2_dir + '/' + img_name[0] + '.png', outputs2_label[0])

        from utils.SCDD_eval import Eval_preds
        _, _, score = Eval_preds(infer_list, label_list)
        print("score is %.6f" % score)
        return score

    def inferSenCD2_score(self,mode='_final_iter'):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        '''
        start_time = time.clock()
        self.net.eval()
        mode_snap = mode
        model_path = self.config.model_dir + '/' + self.config.pred_name + mode_snap + '.pth'
        self.net.load_state_dict(torch.load(model_path))
        #,self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络

        if self.config.dataset_name=="sensetime":
            change_type = ['0_0',
                           '1_2', '1_3', '1_4', '1_5', '1_6',
                           '2_1', '2_3', '2_4', '2_5', '2_6',
                           '3_1', '3_2', '3_4', '3_5', '3_6',
                           '4_1', '4_2', '4_3', '4_5', '4_6',
                           '5_1', '5_2', '5_3', '5_4', '5_5', '5_6',
                           '6_1', '6_2', '6_3', '6_4', '6_5']#32
            rgb_table={'0':(255,255,255),'1':(0,0,255),'2':(128,128,128),'3':(0,128,0),
                   '4':(0,255,0),'5':(128,0,0),'6':(255,0,0)}
        else:
            change_type = ['0_0',
                           '1_2',  '1_4', '1_5',
                           '2_1', '2_3', '2_5',
                           '3_1', '3_2',
                           '5_1', '5_2', '5_4',
                           ]#12
            rgb_table = {'0': (255, 255, 255), '1': (178, 34, 34), '2': (238, 238, 0), '3': (34, 139, 34),
                         '4': (0, 139, 139), '5': (0, 0, 255)}

        infer_list=[]
        label_list=[]

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):

            with torch.no_grad():

                imgs,labels=sample['img'],sample['label']
                img_name=sample['name']
                imgs, labels = imgs.cuda(), labels.cuda()
                print("processing image {}, size is {}".format(i, imgs.shape))

                #outputs = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                outputs=self.net(imgs)

                outputs0 = torch.argmax(outputs, dim=1)#[4,32,256,256]==>[4,256,256]
                outputs0 = outputs0.data.cpu().numpy().astype('uint8')

                #batch_size=masks_pred.shape[0]
                for b in range(outputs0.shape[0]):
                    masks_pred=outputs0[b,...]
                    masks_label=labels[b,...]


                    pred1 = np.zeros((masks_pred.shape), dtype='uint8')
                    pred2 = np.zeros((masks_pred.shape), dtype='uint8')

                    label1 = np.zeros((masks_pred.shape), dtype='uint8')
                    label2 = np.zeros((masks_pred.shape), dtype='uint8')

                    for i in range(masks_pred.shape[0]):
                        for j in range(masks_pred.shape[1]):
                            cur_change = change_type[masks_pred[i, j]]
                            idx1 = cur_change[:cur_change.find('_')]
                            idx2 = cur_change[cur_change.find('_') + 1:]

                            pred1[i,j]=idx1
                            pred2[i,j]=idx2

                            cur_change = change_type[masks_label[i, j]]
                            idx1 = cur_change[:cur_change.find('_')]
                            idx2 = cur_change[cur_change.find('_') + 1:]
                            label1[i,j]=idx1
                            label2[i,j]=idx2



                    infer_list.append(pred1)
                    infer_list.append(pred2)
                    label_list.append(label1)
                    label_list.append(label2)
                    #del pred1,pred2,label1,label2

            cv2.imwrite(self.config.pred1_dir+'/'+img_name[0]+'.png',pred1)
            cv2.imwrite(self.config.pred2_dir +'/'+ img_name[0] + '.png', pred2)

            pred1_rgb = np.zeros((pred1.shape + (3,)), dtype='uint8')
            pred2_rgb = np.zeros((pred2.shape + (3,)), dtype='uint8')

            for i in range(pred1.shape[0]):
                for j in range(pred1.shape[1]):

                    key1 = str(pred1[i, j])
                    key2 = str(pred2[i, j])
                    for k in range(3):
                        pred1_rgb[i, j, 2 - k] = rgb_table[key1][k]  # opencv should write in BGR mode
                        pred2_rgb[i, j, 2 - k] = rgb_table[key2][k]
            cv2.imwrite(self.config.pred1_rgb_dir + '/' + img_name[0] + '.png', pred1_rgb)
            cv2.imwrite(self.config.pred2_rgb_dir + '/' + img_name[0] + '.png', pred2_rgb)


            del pred1, pred2, label1, label2
        from utils.SCDD_eval import Eval_preds
        IoU_mean, Sek, Score = Eval_preds(infer_list, label_list)

        end_time = time.clock()
        run_time = (end_time - start_time)
        mylog = open(self.config.precision_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        print("model is %s" % self.config.model_name)
        print("prediction time is %fs" % run_time)
        print("prediction mIOU is %.6f" % IoU_mean)
        print("prediction Sek is %.6f" % Sek)
        print("prediction Score is %.6f" % Score)

        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup

        print("prediction mIOU is %.6f" % IoU_mean)
        print("prediction Sek is %.6f" % Sek)
        print("prediction Score is %.6f" % Score)

        return Score
















    def inferSenCD(self,use_ave=False,mode='_best_acc'):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        '''

        self.net.eval()
        mode_snap = mode
        model_path = self.config.model_dir + '/' + self.config.pred_name + mode_snap + '.pth'
        self.net.load_state_dict(torch.load(model_path))
        #self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络
        image_size=self.config.patch_size
        target_test=[]
        pred_test=[]


        precision = 0.0
        recall = 0.0
        acc = 0
        f1_score= 0.0
        #pred_batch = 16
        pred_batch=51
        test_num = len(self.testDataloader)/pred_batch
        test_mod=len(self.testDataloader)%pred_batch
        test_batch=pred_batch*test_num

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):

            imgs=sample['img']
            img_name=sample['name']#img_name[0]
            label_test=sample['label']

            label_test = label_test[0].squeeze(0).data.numpy()#[512,512]
            #label_test=label_test.squeeze(0).data.numpy()#[1,512,512]
            if self.cuda:
                imgs= imgs.cuda(self.gpuID)#[1,6,512,512]
                #imgs=imgs.unsqueeze(0)#for batch_size=1

            with torch.no_grad():

                # featT1, featT2 = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                # dist = F.pairwise_distance(featT1, featT2, keepdim=True)
                # dist = F.interpolate(dist, size=imgs.shape[2:], mode='bilinear', align_corners=True)
                # masks_pred = (dist > 1).float()
                if self.config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU" or self.config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU_DCN" or self.config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU_DCN2" or self.config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU_Flow":
                    if self.config["network_G_CD"]["use_DS"]:
                        _, _, _, masks_pred = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        masks_pred = self.net(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                else:
                    masks_pred = self.net(imgs)#for FC-EF


                masks_pred = masks_pred[0].data.cpu().numpy()

                #print("processing image size is:",masks_pred.shape)
                print("processing image {}, size is {}".format(i,masks_pred.shape))
                #masks_pred=self.net(imgs)#输入必须是训练集的整数倍
                predict_img=self.grayTrans_numpy(masks_pred)
                predict_img.save('%s/%s.png'% (self.config.pred_dir,img_name[0]))
                #pred_test_p+=[masks_pred[0]]
                predict_img = np.array(predict_img).astype('uint8')
                _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)#set thresh=90 to fill the fake boundary caused by inaccurate labeling
                cv2.imwrite('%s/%s%s.png'% (self.config.pred_dir,'/Binary/',img_name[0]), binary_img)
                # =======remove hole===============
                save_dir = self.config.pred_dir + '/remove_hole_area'
                mkdir_if_not_exist(save_dir)
                res_img = post_proc(binary_img)
                cv2.imwrite(save_dir + '/' + img_name[0] + '.png', res_img)

                res_img = np.array(res_img).astype('float')
                res_img /= 255#=====================for acc evaluation, y_pred must be [0,1]
                #pred_test+=[res_img]#=============for accuracy evaluation

                binary_img = np.array(binary_img).astype('float')  # for raw binary
                binary_img /= 255
                if use_ave == False:
                   #pred_test += [binary_img]
                   pred_test += [res_img]
                   #pred_test_p += [masks_pred[0]]
                   target_test += [label_test]
                else:
                   pred_test += [binary_img]
                   target_test += [label_test]
                   if i%pred_batch==0:
                       _precision, _recall, _f1_score, _acc = self.PR_score_whole(target_test, pred_test)
                       precision += _precision
                       recall += _recall
                       f1_score += _f1_score
                       acc += _acc
                       pred_test=[]
                       target_test=[]
                   if i==len(self.testDataloader) and test_mod>0:
                       _precision, _recall, _f1_score, _acc = self.PR_score_whole(target_test, pred_test)
                       precision += _precision
                       recall += _recall
                       f1_score += _f1_score
                       acc += _acc
                       test_num+=1
        if use_ave:
            return precision*1.0/test_num,recall*1.0/test_num,f1_score*1.0/test_num,acc*1.0/test_num

        return  np.array(target_test),np.array(pred_test)



    def inferSenBR(self,multi_outputs=False,use_ave=False,use_TTA=True,use_scaleATT=False):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        '''
        #self.model_path=r'E:\TestData\Building&Road\WHU-Building\Building_aerial\result\model\netG_AS_Ref_UNet_2D_patch_512_batch_4_nepoch_2025980_G.pth'
        #self.model_path = 'E:/TestData/Building&Road/DeepGlobe/road-train-2+valid.v2/patch256/train80/result/model/unet3+_stem_res_n3_dense_bce_ssim_deepsub_ref32_20000.pth'
        #self.model_path='E:/TestData/Building&Road/Ottawa-Dataset/Ottawa-Dataset/patch256/train80/result/model/unet3+_stem_res2_dense_bce_ssim_20000.pth'
        #self.model_path=r'E:\TestData\Building&Road\WHU-Building\Building_aerial\result\model\netG_UNet_3Plus_patch_512_batch_4_nepoch_2025980_G.pth'
        #self.model_path=r'E:\TestData\Building&Road\WHU-Building\Building_aerial\result\model//netG_PreGAUE_unet_2D_PreTrainSC_patch_512_batch_4_nepoch_2025980_G.pth'
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络
        image_size=self.config.patch_size
        target_test=[]
        pred_test=[]
        pred_test_p=[]

        precision = 0.0
        recall = 0.0
        acc = 0
        f1_score= 0.0
        #pred_batch = 16
        pred_batch=51
        test_num = len(self.testDataloader)/pred_batch
        test_mod=len(self.testDataloader)%pred_batch
        test_batch=pred_batch*test_num

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):

            imgs=sample['img']
            img_name=sample['name']#img_name[0]
            label_test=sample['label']
            # label_path = self.test_dir + '/label/' + img_name[0]+".png"
            # label_path1 = self.test_dir + '/label/' + img_name[0] + ".jpg"
            # label_path2 = self.test_dir + '/label/' + img_name[0] + ".tif"
            # try:
            #      label_test = np.asarray(Image.open(label_path).convert('L'))
            # except:
            #     try:
            #         label_test = np.asarray(Image.open(label_path1).convert('L'))
            #     except:
            #         label_test = np.asarray(Image.open(label_path2).convert('L'))
            #
            # label_test =label_test.astype('float')
            # label_test /= 255
            #target_test+=[label_test.squeeze(0).data.numpy()]
            label_test=label_test.squeeze(0).data.numpy()
            if self.cuda:
                imgs= imgs.cuda(self.gpuID)
                imgs=imgs.unsqueeze(0)#for batch_size=1

            with torch.no_grad():
                if self.multi_outputs:
                   #masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_xy(xx, self.net))#==>numpy

                   #masks_pred,_,_,_,_,_=self.net(imgs)
                   #========for multi-scale output===========
                   # preds0,preds1,preds2,preds3=self.net(imgs)
                   # masks_pred=(preds0+preds1+preds2+preds3)/4
                   #==========for MC output============
                   # outputs0, outputs1 = self.net(imgs)
                   # pred_prob = F.softmax(outputs0 + outputs1, dim=1)
                   # masks_pred = pred_prob[:, 1].unsqueeze(1)
                   #===============for multi-output====================
                 if self.config["network_G"]["out_nc"]>1:
                     outputs0, outputs1 = self.net(imgs)
                     pred_prob = F.softmax(outputs1, dim=1)
                     masks_pred = pred_prob[:, 1].unsqueeze(1)
                 else:
                     if self.config["network_G"]["which_model_G"] == 'EDCls_UNet_BCD':
                         _,_,_,masks_pred = self.net(imgs, imgs, use_warm=True)


                 masks_pred = masks_pred[0].data.cpu().numpy()
                else:

                   #masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_x(xx, self.net))
                   #=====imgs = torch.unsqueeze(imgs, dim=-1)#for unet_3D not necessary
                   if use_TTA:
                       if use_scaleATT:
                           out_logits0, out_logits1, att_map0 = self.net(imgs)
                           out_seg0 = F.interpolate(out_logits0 * att_map0, scale_factor=2, mode='bilinear',
                                                    align_corners=True)
                           att_map0 = F.interpolate(att_map0, scale_factor=2, mode='bilinear', align_corners=True)
                           out_seg1 = out_logits1 * (1 - att_map0)
                           masks_pred = F.sigmoid(out_seg0 + out_seg1)
                       else:
                           masks_pred = self.predict_TTA(self.net, imgs)

                       print("using TTA...")
                   else:
                       if use_scaleATT:
                           out_logits0, out_logits1, att_map = self.net(imgs)
                           att_map0 = att_map[:, 0].unsqueeze(1)
                           att_map1 = att_map[:, 1].unsqueeze(1)
                           out_seg = out_logits0 * att_map0 + out_logits1 * att_map1
                           masks_pred = F.sigmoid(out_seg)
                       else:
                           masks_pred = self.net(imgs)

                   masks_pred = masks_pred[0].data.cpu().numpy()

                #print("processing image size is:",masks_pred.shape)
                print("processing image {}, size is {}".format(i,masks_pred.shape))
                #masks_pred=self.net(imgs)#输入必须是训练集的整数倍
                predict_img=self.grayTrans_numpy(masks_pred)
                predict_img.save('%s/%s.png'% (self.pred_dir,img_name))
                #pred_test_p+=[masks_pred[0]]
                predict_img = np.array(predict_img).astype('uint8')
                _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)#set thresh=90 to fill the fake boundary caused by inaccurate labeling
                cv2.imwrite('%s/%s%s.png'% (self.pred_dir,'/Binary/',img_name), binary_img)
                # =======remove hole===============
                save_dir = self.pred_dir + '/remove_hole_area'
                mkdir_if_not_exist(save_dir)
                res_img = post_proc(binary_img)
                cv2.imwrite(save_dir + '/' + img_name + '.png', res_img)

                res_img = np.array(res_img).astype('float')
                res_img /= 255#=====================for acc evaluation, y_pred must be [0,1]
                #pred_test+=[res_img]#=============for accuracy evaluation

                binary_img = np.array(binary_img).astype('float')  # for raw binary
                binary_img /= 255
                if use_ave == False:
                   pred_test += [binary_img]
                   pred_test_p += [masks_pred[0]]
                   target_test += [label_test]
                else:
                   pred_test += [binary_img]
                   target_test += [label_test]
                   if i%pred_batch==0:
                       _precision, _recall, _f1_score, _acc = self.PR_score_whole(target_test, pred_test)
                       precision += _precision
                       recall += _recall
                       f1_score += _f1_score
                       acc += _acc
                       pred_test=[]
                       target_test=[]
                   if i==len(self.testDataloader) and test_mod>0:
                       _precision, _recall, _f1_score, _acc = self.PR_score_whole(target_test, pred_test)
                       precision += _precision
                       recall += _recall
                       f1_score += _f1_score
                       acc += _acc
                       test_num+=1








        if use_ave:
            return precision*1.0/test_num,recall*1.0/test_num,f1_score*1.0/test_num,acc*1.0/test_num

        return  np.array(target_test),np.array(pred_test),np.array(pred_test_p)


    def inferSen(self,multi_inputs=False):
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络
        image_size=self.config.image_size
        target_test=[]
        pred_test=[]

        for i, sample in enumerate(self.testDataloader, 0):
             #if multi_inputs==False:
            imgs=sample['image']
            img_name=sample['name']
            label_path = self.test_dir + '/label/' + img_name[0]+".tif"
            label_test= np.asarray(Image.open(label_path).convert('L'))
            label_test =label_test.astype('float')
            label_test /= 255
            target_test+=[label_test]
            if self.cuda:
                imgs= imgs.cuda(self.gpuID)

            with torch.no_grad():
                if multi_inputs:
                   masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_xy(xx, self.net))#==>numpy
                else:

                   #masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_x(xx, self.net))
                   #=====imgs = torch.unsqueeze(imgs, dim=-1)#for unet_3D not necessary
                   masks_pred = self.predict_img_pad(imgs, image_size, lambda xx: self.predict_x(xx, self.net))


                #print("processing image size is:",masks_pred.shape)
                print("processing image {}, size is {}".format(i,masks_pred.shape))
                #masks_pred=self.net(imgs)#输入必须是训练集的整数倍
                predict_img=self.grayTrans_numpy(masks_pred)
                predict_img.save('%s/%s.png'% (self.pred_dir,img_name[0]))
                predict_img = np.array(predict_img).astype('uint8')
                _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite('%s/%s%s.png'% (self.pred_dir,'/Binary/',img_name[0]), binary_img)
                # =======remove hole===============
                save_dir = self.pred_dir + '/remove_hole_area'
                mkdir_if_not_exist(save_dir)
                res_img = post_proc(binary_img)
                cv2.imwrite(save_dir + '/' + img_name[0] + '.tif', res_img)

                res_img = np.array(res_img).astype('float')
                res_img /= 255#=====================for acc evaluation, y_pred must be [0,1]
                pred_test+=[res_img]#=============for accuracy evaluation
                # ==============================res img 1 nochange 2 change========================
                # res_img = np.array(res_img).astype('float')
                # res_img /= 255
                # res_img += 1
                # res_img = np.array(res_img).astype('uint8')
                # save_dir = self.pred_dir + '/Upload'
                # mkdir_if_not_exist(save_dir)
                # cv2.imwrite(save_dir + '/' + img_name[0] + '.tif', res_img)
        return  np.array(target_test),np.array(pred_test)

    def infer(self):
        self.model_path='E:/TestData/Building&Road/DeepGlobe/road-train-2+valid.v2/patch256/train80/result/model/unet3+_stem_res_n3_dense_bce_ssim_deepsub_ref32_20000.pth'
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path))#通过网络参数形式加载网络
        correctsAcc = 0
        # perform test inference
        #with torch.no_grad:
        for i, sample in enumerate(self.testDataloader, 0):
            # get the test sample
             imgs, true_masks = sample

             if self.cuda:
                imgs, true_masks = imgs.cuda(self.gpuID), true_masks.cuda(self.gpuID)
             imgs, true_masks = Variable(imgs), Variable(true_masks)  # data-->data+grad
             masks_pred = self.net(imgs)# 也可以 self.net.forward(imgs)
             # masks_probs_flat = masks_pred.view(-1)
             # true_masks_flat = true_masks.view(-1)
             # masks_probs_flat = masks_probs_flat > 0.5
             # correctsAcc += torch.sum(
             #        masks_probs_flat.int() == true_masks_flat.int()).item() * 1.0 / true_masks_flat.size(0)
             correctsAcc=Acc(masks_pred,true_masks)
             if self.batchsize==1:
                predict_img=self.grayTrans(masks_pred)
                predict_img.save('%s/%d.png'% (self.pred_dir,i))
                print("writing %d img" % i)

                predict_img = np.array(predict_img).astype('uint8')
                _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite('%s/%s%d.png'% (self.pred_dir,'/Binary/',i), binary_img)


             else:
                 for img_idx in range(self.batchsize):
                     predict_img = self.grayTrans(masks_pred.data.cpu().numpy()[img_idx],use_batch=True)
                     predict_img.save('%s/%d.png' % (self.pred_dir, img_idx+i*self.batchsize))

                     predict_img = np.array(predict_img).astype('uint8')
                     _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)
                     cv2.imwrite('%s/%s%d.png' % (self.pred_dir, '/Binary/', img_idx+i*self.batchsize), binary_img)
                 print("writing %d batch" % i)
        test_acc = correctsAcc * 1.0 / len(self.testDataloader)
        return  test_acc


    def computeAcc(self):
        mylog = open(self.config.precison_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        test_acc = self.infer()
        print("prediction acc is %.6f" % test_acc)
        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup



