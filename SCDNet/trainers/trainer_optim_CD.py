import math
import os, time,sys
import numpy as np
from PIL import Image
import os.path as osp

# import torch modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_loaders.RSCD_dl import RSCD_DL
from torchsummary import summary
import matplotlib.pyplot as plt

import logging
from losses.myLoss import bce_edge_loss
from utils.utils import PR_score_whole

from tqdm import tqdm



class TrainerOptim(object):
    # init function for class
    def __init__(self, config,trainDataloader, valDataloader,trainDataloader2=None
                ):
        dl=RSCD_DL(config)
        self.config=dl.config
        self.model_path=dl.config.model_name
        self.log_file=dl.config.log_path
        self.lossImg_path=dl.config.loss_path

        # set the GPU flag

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.valDataloader = valDataloader
        self.trainDataloader = trainDataloader
        self.trainDataloader2 = trainDataloader2

        self.pix_cri=bce_edge_loss(use_edge=True).to(self.device)
        self.pix_cri0 = bce_edge_loss(use_edge=False).to(self.device)
        from models.Satt_CD.modules.loss import BCL
        self.cri_dist = BCL().to(self.device)


    def load_ck(self,model,model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        loaded_dict = checkpoint
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)


    def train_optim(self):
        # create model
        start_time = time.clock()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger
        #model = create_model(self.config)
        # resume state??
        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        # setup_logger('val', self.config.log_dir, 'val_'+self.config.pred_name, level=logging.INFO)
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"loss": [],
                         "acc":[],
                         "val_loss": [],
                         "val_acc": [],
                         }
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * len(self.trainDataloader))
        self.config['train']['niter']=total_iters
        self.config["train"]["lr_steps"] = [int(0.25 * total_iters), int(0.5 * total_iters), int(0.75 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters) - 1]
        model = create_model(self.config)  # create model after updating config
        #multi_outputs = self.config["network_G-CD"]["multi_outputs"]
        best_acc = 0
        best_loss=1000
        # if self.config["network_G_CD"]["which_model_G"]=='EDCls_Net':
        #    self.load_ck(model.netG,self.config.pretrained_model_path)#loading model checkpoint
        if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':

           self.load_ck(model.netCD, self.config.pretrained_model_path)  # loading model checkpoint
           model.netCD.eval()

        use_warmup=True if self.config["train"]["warmup_epoch"]>0 else False
        if use_warmup:
            logger.info("using warmup for training")
            for optim in model.optimizers:
                optim.zero_grad()
                optim.step()
        if self.config["train"]["use_progressive_resize"]:
            logger.info("using progressive resize for training")


        for epoch in range(0, total_epochs):
            print('Epoch {}/{}'.format(epoch + 1, total_epochs))
            print('-' * 60)
            epoch_loss = 0
            epoch_acc = 0

            if use_warmup or self.config["train"]["lr_scheme"]=="CosineLR":
                model.update_learning_rate()
            cur_Dataloader = self.trainDataloader





            for i, sample in enumerate(tqdm(cur_Dataloader, 0)):
                current_step += 1
                # training
                model.feed_data(sample)
                model.netG.train()

                #model.optimize_parameters(current_step)
                if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3':#for ISCD

                    model.optimize_parameters_MC7_rgb255(current_step)
                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_Res50':

                    model.optimize_parameters_MC7_DS(current_step)

                elif self.config['network_G_CD']['which_model_G'] == 'FC_EF' or self.config["network_G_CD"]["which_model_G"]=="Seg_EF":
                    model.optimize_parameters_SC1(current_step)#single input  CD


                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin'or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin':
                    model.optimize_parameters_MC7_bin(current_step)

                else:
                    if self.config["train"]["use_CatOut"]:
                        model.optimize_parameters_MC7_DS(current_step)
                    else:
                        model.optimize_parameters_MC7(current_step)

                #model.optimize_parameters_MC7_rgb255(current_step)
                #model.optimize_parameters_MC7_DS(current_step)#for cat_out

                # update learning rate
                if use_warmup==False and self.config["train"]["lr_scheme"]=="MultiStepLR":
                   model.update_learning_rate()

                if current_step % self.config['logger']['print_freq'] == 0:
                    logs = model.get_current_log()
                    epoch_loss += logs['l_g_total']
                    #epoch_acc += logs['pnsr']
                    # message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}, lossD_grad:{:.6f}> '.format(
                    #     epoch, current_step, model.get_current_learning_rate(),
                    #     logs['l_g_total'],
                    #     0.5*(logs['l_d_real']+logs['l_d_fake']),0.5*(logs['l_d_real_grad']+logs['l_d_fake_grad']))

                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                        epoch, current_step, model.get_current_learning_rate(use_warmup=use_warmup),
                        logs['l_g_total'],logs['l_d_total']

                    )
                    logger.info(message)
                    #=======for val test======================================
                    # val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                    # #val_loss, val_acc = self.val(epoch + 1, model=model)
                    # message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                    # logger.info(message)
                    #=========================================================
                if current_step in self.config['logger']['save_iter']:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
            if epoch % self.config['train']['val_epoch'] == 0:
                # if self.config["network_G_CD"]["which_model_G"] == 'EDCls_Net2' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet' or \
                #     self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet4'\
                #         or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_DiffAdd' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet5' \
                #         or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5'\
                #         or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_Res50':
                #     val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                # else:
                #     val_loss, val_acc = self.val(epoch + 1, model=model, multi_outputs=True)
                if self.config["network_G_CD"]["out_nc"]>1:
                    if self.config["network_G_CD"]["out_nc"]>10:
                        #val_loss, val_acc = self.val_SEK32(epoch + 1, model=model)
                        val_loss, val_acc =0,0
                    else:
                        val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)

                else:
                    val_loss, val_acc = self.val(epoch + 1, model=model)

                message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                logger.info(message)
                logs = model.get_current_log()
                #train_history["loss"].append(logs['l_g_total'])
                train_history["loss"].append(epoch_loss * 1.0 / len(self.trainDataloader))
                #train_history["acc"].append(epoch_acc * 1.0 / len(self.trainDataloader))
                train_history["val_loss"].append(val_loss)
                train_history["val_acc"].append(val_acc)
                if self.config["network_G_CD"]["out_nc"] < 10:
                    if val_acc > best_acc:
                        best_acc = val_acc
                        model.save_best_acc()
                    if val_loss < best_loss:
                        best_loss = val_loss
                        model.save_best_loss()



        end_time = time.clock()
        run_time=end_time-start_time
        #print(end_time - start_time, 'seconds')
        message='running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim(train_history)

    def train_optim_tune(self):

        # create model
        start_time = time.clock()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger

        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        # setup_logger('val', self.config.log_dir, 'val_'+self.config.pred_name, level=logging.INFO)
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"loss": [],
                         "acc":[],
                         "val_loss": [],
                         "val_acc": [],
                         }
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * len(self.trainDataloader))
        self.config['train']['niter']=total_iters
        self.config["train"]["lr_steps"] = [int(0.25 * total_iters), int(0.5 * total_iters), int(0.75 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters) - 1]
        model = create_model(self.config)  # create model after updating config
        #multi_outputs = self.config["network_G-CD"]["multi_outputs"]
        best_acc = 0
        best_loss=1000

        self.load_ck(model.netG,self.config.pretrained_model_path)#loading model checkpoint
        use_warmup = True if self.config["train"]["warmup_epoch"] > 0 else False
        #=======frozon feature param=======
        for p in model.netG.feat_Extactor.parameters():
            p.requires_grad = False

        # use_warmup=True if self.config["train"]["warmup_epoch"]>0 else False
        # if use_warmup:
        #     logger.info("using warmup for training")
        #     for optim in model.optimizers:
        #         optim.zero_grad()
        #         optim.step()

        logger.info("using fine-tune for training,frozen the params of the feat_extrator...")
        for epoch in range(0, total_epochs):
            print('Epoch {}/{}'.format(epoch + 1, total_epochs))
            print('-' * 60)
            epoch_loss = 0
            epoch_acc = 0

            # if use_warmup:
            #     model.update_learning_rate()
            if use_warmup or self.config["train"]["lr_scheme"]=="CosineLR":
                model.update_learning_rate()#update lr pr epoch for consineLR




            cur_Dataloader = self.trainDataloader

            for i, sample in enumerate(tqdm(cur_Dataloader, 0)):
                current_step += 1
                # training
                model.feed_data(sample)
                model.netG.train()
                model.optimize_parameters_MC7(current_step)


                # update learning rate
                #if use_warmup==False:
                #model.update_learning_rate()
                if use_warmup==False and self.config["train"]["lr_scheme"]=="MultiStepLR":
                   model.update_learning_rate()

                if current_step % self.config['logger']['print_freq'] == 0:
                    logs = model.get_current_log()
                    epoch_loss += logs['l_g_total']

                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                        epoch, current_step, model.get_current_learning_rate(use_warmup=False),
                        logs['l_g_total'],logs['l_d_total']

                    )
                    logger.info(message)
                    #=======for val test======================================
                    # val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                    # message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                    # logger.info(message)
                    #=========================================================
                if current_step in self.config['logger']['save_iter']:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
            if epoch % self.config['train']['val_epoch'] == 0:

                val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)

                message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                logger.info(message)
                logs = model.get_current_log()
                #train_history["loss"].append(logs['l_g_total'])
                train_history["loss"].append(epoch_loss * 1.0 / len(self.trainDataloader))
                #train_history["acc"].append(epoch_acc * 1.0 / len(self.trainDataloader))
                train_history["val_loss"].append(val_loss)
                train_history["val_acc"].append(val_acc)
                if val_acc> best_acc:
                    best_acc = val_acc
                    model.save_best_acc()
                if val_loss<best_loss:
                    best_loss=val_loss
                    model.save_best_loss()



        end_time = time.clock()
        run_time=end_time-start_time
        #print(end_time - start_time, 'seconds')
        message='running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim(train_history)



    def train_optim_cos(self):
        # create model
        start_time = time.clock()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger
        # model = create_model(self.config)
        # resume state??
        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        # setup_logger('val', self.config.log_dir, 'val_'+self.config.pred_name, level=logging.INFO)
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"loss": [],
                         "acc": [],
                         "val_loss": [],
                         "val_acc": [],
                         }
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * len(self.trainDataloader))
        self.config['train']['niter'] = total_iters
        self.config["train"]["lr_steps"] = [int(0.25 * total_iters), int(0.5 * total_iters), int(0.75 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters) - 1]
        model = create_model(self.config)  # create model after updating config
        multi_outputs = self.config["network_G"]["multi_outputs"]
        best_acc = 0
        best_loss = 1000
        if self.config["network_G_CD"] == 'EDCls_Net':
            self.load_ck(model.netG, self.config.pretrained_model_path)  # loading model checkpoint

        init_lr = self.config["train"]["lr_G"]
        epochs_per_cycle = total_epochs // self.config["train"]["cos_cycle"]
        model_shots = []
        for i_cycle in range(self.config["train"]["cos_cycle"]):
            for epoch in range(epochs_per_cycle):
                print('Cycle_{},Epoch {}/{}'.format(i_cycle,epoch + 1, epochs_per_cycle))
                print('-' * 60)
                epoch_loss = 0
                epoch_acc = 0
                cur_lr=model.update_learning_rate_cos(init_lr, epoch, epochs_per_cycle)

                for i, sample in enumerate(tqdm(self.trainDataloader, 0)):
                    current_step += 1
                    # training
                    model.feed_data(sample)
                    model.netG.train()
                    model.optimize_parameters_MC7(current_step)

                    if current_step % self.config['logger']['print_freq'] == 0:
                        logs = model.get_current_log()
                        epoch_loss += logs['l_g_total']

                        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                            epoch, current_step, cur_lr,
                            logs['l_g_total'], logs['l_d_total']
                        )
                        logger.info(message)
                        # =======for val test======================================
                        # val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                        # message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                        # logger.info(message)
                        # =========================================================
                    # if current_step in self.config['logger']['save_iter']:
                    #     logger.info('Saving models and training states.')
                    #     model.save(current_step)

                if epoch % self.config['train']['val_epoch'] == 0:
                    if self.config["network_G_CD"]["which_model_G"] == 'EDCls_Net2' or self.config["network_G_CD"][
                        "which_model_G"] == 'EDCls_UNet' or \
                            self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2' or \
                            self.config["network_G_CD"][
                                "which_model_G"] == 'EDCls_UNet3' or self.config["network_G_CD"][
                        "which_model_G"] == 'EDCls_UNet4'or self.config["network_G_CD"][
                        "which_model_G"] == 'EDCls_UNet2_DiffAdd':
                        val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                    else:
                        val_loss, val_acc = self.val(epoch + 1, model=model, multi_outputs=multi_outputs)

                    message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                    logger.info(message)
                    logs = model.get_current_log()
                    train_history["loss"].append(epoch_loss * 1.0 / len(self.trainDataloader))
                    train_history["val_loss"].append(val_loss)
                    train_history["val_acc"].append(val_acc)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        model.save_best_acc_cycle(i_cycle)


        end_time = time.clock()
        run_time = end_time - start_time
        message = 'running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim(train_history)




    def visualize_train_optim(self, history):

        val_acc = history["val_acc"]
        loss = history["loss"]
        acc=history["acc"]
        val_loss = history["val_loss"]
        plt.subplot(121)
        #plt.plot(acc)#for the acc is much accurate to calculate using large batch, we currently do not compute it for each batch
        plt.plot(val_acc)
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['valid'], loc='upper left')
        plt.subplot(122)
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper right')

        plt.savefig(self.lossImg_path)
        plt.show()
    def visualize_train(self,history):
        acc = history["acc"]
        val_acc = history["val_acc"]
        loss = history["loss"]
        val_loss = history["val_loss"]
        plt.subplot(121)
        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('model pnsr')
        plt.ylabel('pnsr')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.subplot(122)
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper right')

        plt.savefig(self.lossImg_path)
        plt.show()

    def print_info(self,history={},elapse_time=0.0,epochs=20):
        mylog = open(self.log_file, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog  # 输出到文件


        print(summary(self.net,(3, 48, 48)))

        print("model train time is %.6f s" % elapse_time)
        print('model_name:', self.model_path)
        loss=history['loss']# equal to history["loss"]
        acc=history["acc"]
        val_loss = history["val_loss"]
        val_acc = history["val_acc"]
        for i in range(epochs):
            print('epoch: %d' % (i + 1))
            print('train_loss: %.5f' % loss[i], 'val_loss:%.5f' % val_loss[i])
            print('train_acc:%.5f' % acc[i], 'val_acc:%.5f' % val_acc[i])
            mylog.flush()
        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup

    def save_checkpoint(self,state, is_best, filename=None):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print("==> Saving a new best")
            torch.save(state, filename)  # save checkpoint
        else:
            print("==> Validation Accuracy did not improve")


    def  val_SEK32(self, epoch,model=None,multi_outputs=False):
        # eval model on validation set

        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        infer_list=[]
        label_list=[]
        # perform test inference
        val_model = model.netG
        val_model.eval()

        change_type = ['0_0',
                       '1_2', '1_3', '1_4', '1_5', '1_6',
                       '2_1', '2_3', '2_4', '2_5', '2_6',
                       '3_1', '3_2', '3_4', '3_5', '3_6',
                       '4_1', '4_2', '4_3', '4_5', '4_6',
                       '5_1', '5_2', '5_3', '5_4', '5_5', '5_6',
                       '6_1', '6_2', '6_3', '6_4', '6_5']

        rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                     '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}
        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():

                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()

                #outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                outputs = val_model(imgs)
                outputs0 = torch.argmax(outputs, dim=1)#[4,32,256,256]==>[4,256,256]
                outputs0 = outputs0.data.cpu().numpy().astype('uint8')

                #batch_size=masks_pred.shape[0]
                for b in range(outputs0.shape[0]):
                    masks_pred=outputs0[b,...]
                    masks_label=labels[b,...]
                    # pred1 = np.zeros((masks_pred.shape + (3,)), dtype='uint8')
                    # pred2 = np.zeros((masks_pred.shape + (3,)), dtype='uint8')
                    #
                    # label1 = np.zeros((masks_pred.shape + (3,)), dtype='uint8')
                    # label2 = np.zeros((masks_pred.shape + (3,)), dtype='uint8')

                    pred1 = np.zeros((masks_pred.shape), dtype='uint8')
                    pred2 = np.zeros((masks_pred.shape), dtype='uint8')

                    label1 = np.zeros((masks_pred.shape), dtype='uint8')
                    label2 = np.zeros((masks_pred.shape), dtype='uint8')

                    for i in range(masks_pred.shape[0]):
                        for j in range(masks_pred.shape[1]):
                            cur_change = change_type[masks_pred[i, j]]
                            idx1 = cur_change[:cur_change.find('_')]
                            idx2 = cur_change[cur_change.find('_') + 1:]
                            # key1 = str(idx1)
                            # key2 = str(idx2)
                            pred1[i,j]=idx1
                            pred2[i,j]=idx2

                            cur_change = change_type[masks_label[i, j]]
                            idx1 = cur_change[:cur_change.find('_')]
                            idx2 = cur_change[cur_change.find('_') + 1:]
                            label1[i,j]=idx1
                            label2[i,j]=idx2

                            # key1_label = str(idx1)
                            # key2_label = str(idx2)

                            #for k in range(3):
                                # pred1[i, j, k] = rgb_table[key1][k]
                                # pred2[i, j, k] = rgb_table[key2][k]
                                # label1[i, j, k] = rgb_table[key1_label][k]
                                # label2[i, j, k] = rgb_table[key2_label][k]

                    infer_list.append(pred1)
                    infer_list.append(pred2)
                    label_list.append(label1)
                    label_list.append(label2)
                    del pred1,pred2,label1,label2




                ce_loss = nn.CrossEntropyLoss()
                loss = ce_loss(outputs, labels)


                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()


        from utils.SCDD_eval import Eval_preds
        _,_,score=Eval_preds(infer_list,label_list)

        val_loss=lossAcc*1.0/len(self.valDataloader)
        val_acc=score

        print('Epoch %d evaluate done ' % epoch)
        # convert to train mode for next training



        return  val_loss,val_acc



    def  val_SEK7(self, epoch,model=None):
        # eval model on validation set

        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        infer_list=[]
        label_list=[]
        # perform test inference
        val_model = model.netG
        val_model.eval()

        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():
                # imgs_LR = sample['LR']
                # imgs_HR=sample['HR']
                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                gt1_label=labels[:,0,:,:].data.cpu().numpy().astype('uint8')
                gt2_label = labels[:, 1, :, :].data.cpu().numpy().astype('uint8')

                if self.config["network_G_CD"]["use_DS"]:
                    #if self.config["train"]["use_label_rgb255"]:
                    if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3':#for rgb255 guidance
                        pred1, pred2, pred3, (outputs1, outputs2,_) = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    #elif  self.config["train"]["use_CatOut"]:
                    elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5':
                        pred1, pred2, pred3,pred4, (outputs1, outputs2) = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':
                        model.netCD.eval()
                        images_T1, images_T2 = imgs[:, 0:3, ...], imgs[:, 3:6, ...]
                        with torch.no_grad():
                            _, _, _, images_cd = model.netCD(images_T1, images_T2)

                        images_T1 = torch.cat([images_T1, images_cd], dim=1)
                        images_T2 = torch.cat([images_T2, images_cd], dim=1)
                        outputs1, outputs2 = val_model(images_T1, images_T2)
                    elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin':
                        pred1, pred2, pred3, (outputs1, outputs2,outputs12) = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    elif  self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                        outputs1, outputs2, outputs12 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        #_, _, _, (outputs1, outputs2) = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                        pred1,pred2,pred3,(outputs1, outputs2)= val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])

                else:
                    if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3':
                        outputs1, outputs2,_ = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    elif  self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                        outputs1, outputs2, outputs12 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        outputs1, outputs2 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])

                outputs1_label = torch.argmax(outputs1, dim=1)#[4,32,256,256]==>[4,256,256]
                outputs1_label = outputs1_label.data.cpu().numpy().astype('uint8')

                outputs2_label=torch.argmax(outputs2, dim=1)
                outputs2_label = outputs2_label.data.cpu().numpy().astype('uint8')

                if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin' or \
                        self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                    outputs12 = outputs12[:, 0, :, :].data.cpu().numpy()
                    if self.config["train"]["use_MC6"]:
                        outputs1_label+=1
                        outputs2_label += 1
                        outputs1_label[outputs12 < 0.5] = 0
                        outputs2_label[outputs12 < 0.5] = 0
                    else:

                        outputs1_label[outputs12<0.5]=0
                        outputs2_label[outputs12<0.5]=0


                infer_list.append(outputs1_label)
                infer_list.append(outputs2_label)
                label_list.append(gt1_label)
                label_list.append(gt2_label)


                #====for loss=============
                if self.config["train"]["use_CatOut"] or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5':
                    loss = self.compute_val_loss_Cat(pred1, pred2, pred3,pred4, (outputs1, outputs2), labels)
                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':
                    loss=self.compute_val_loss_single(outputs1,outputs2,labels)
                elif self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                    # l_g_total = 0
                    labels -= 1
                    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
                    loss = ce_loss(outputs1, labels[:, 0, ...]) + ce_loss(outputs2, labels[:, 1, ...])
                else:
                    #loss=self.compute_val_loss(pred1,pred2,pred3,(outputs1,outputs2),labels)
                    if self.config["network_G_CD"]["use_DS"]:
                        loss=self.compute_val_loss(pred1,pred2,pred3,(outputs1,outputs2),labels)
                    else:
                        loss = self.compute_val_loss_single(outputs1, outputs2, labels)




                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()


        from utils.SCDD_eval import Eval_preds
        _,_,score=Eval_preds(infer_list,label_list)

        val_loss=lossAcc*1.0/len(self.valDataloader)
        val_acc=score

        print('Epoch %d evaluate done ' % epoch)




        return  val_loss,val_acc

    def compute_val_loss(self,pred1,pred2,pred3,pred4,labels):

        # # =========================for sensetime cd==========================================================
        # # class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
        # # class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        #
        # # ======================for franch cd====================
        #
        # class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
        # class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]

        if self.config["dataset_name"] == 'sensetime':
            class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
            class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        else:
            class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
            class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]


        l_g_total=0
        from models.utils import one_hot_cuda
        from models.Satt_CD.modules.loss import ComboLoss
        self.cri_seg_mc = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        self.cri_ce_loss = nn.CrossEntropyLoss().to(self.device)
        class_num = self.config["network_G_CD"]["out_nc"]
        label_smooth = self.config["train"]["use_label_smooth"]

        if self.config["network_G_CD"]["use_DS"]:

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
            labels1, labels2, labels3= labels1.long(), labels2.long(), labels3.long()

            one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels4_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels4_2[:, k, ...])

            ce_weight = self.config["train"]["ce_weight"]
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
        return l_g_total

    def compute_val_loss_Cat(self,pred1,pred2,pred3,pred4,pred5,labels):
        class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
        class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        l_g_total=0
        from models.utils import one_hot_cuda
        from models.Satt_CD.modules.loss import ComboLoss
        self.cri_seg_mc = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        self.cri_ce_loss = nn.CrossEntropyLoss().to(self.device)
        class_num = self.config["network_G_CD"]["out_nc"]
        label_smooth = self.config["train"]["use_label_smooth"]

        if self.config["train"]["use_DS"]:

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
            labels4 = F.interpolate(labels.float(), (img_down_size * 8, img_down_size * 8), mode='bilinear',
                                    align_corners=True)
            labels1, labels2, labels3,labels4= labels1.long(), labels2.long(), labels3.long(),labels4.long()

            one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_1 = one_hot_cuda(labels4[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_2 = one_hot_cuda(labels4[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels5_1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels5_2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels4_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels4_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred5[0][:, k, ...], one_hot_labels5_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred5[1][:, k, ...], one_hot_labels5_2[:, k, ...])

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
        return l_g_total

    def compute_val_loss_single(self,pred1,pred2,labels):
        if self.config["dataset_name"]=='sensetime':
            class_weight1 = [ 0.0007,    0.1970,    0.0065,    0.0085,    0.0391,    0.0176,    0.7307]
            class_weight2 = [0.0011,    0.3595,    0.0154,    0.0265,    0.0524,    0.0104,    0.5347]
        else:
            class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
            class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]
        l_g_total=0
        from models.utils import one_hot_cuda
        from models.Satt_CD.modules.loss import ComboLoss
        self.cri_seg_mc = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        self.cri_ce_loss = nn.CrossEntropyLoss().to(self.device)
        class_num = self.config["network_G_CD"]["out_nc"]
        label_smooth = self.config["train"]["use_label_smooth"]
        one_hot_labels1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
        one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
        ce_weight = self.config["train"]["ce_weight"]
        for k in range(class_num):
            l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
            l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
        l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2,
                                                                                               labels[:, 1,
                                                                                               ...]) * ce_weight

        return l_g_total

    def compute_val_loss_single2(self,pred1,pred2,labels):
        # if self.config["dataset_name"]=='sensetime':
        #     class_weight1 = [ 0.0007,    0.1970,    0.0065,    0.0085,    0.0391,    0.0176,    0.7307]
        #     class_weight2 = [0.0011,    0.3595,    0.0154,    0.0265,    0.0524,    0.0104,    0.5347]
        # else:
        #     class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
        #     class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]
        # l_g_total=0
        # from models.utils import one_hot_cuda
        # from models.Satt_CD.modules.loss import ComboLoss
        # self.cri_seg_mc = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        self.cri_ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        # class_num = self.config["network_G_CD"]["out_nc"]
        # label_smooth = self.config["train"]["use_label_smooth"]
        # one_hot_labels1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
        # one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
        # ce_weight = self.config["train"]["ce_weight"]
        # for k in range(class_num):
        #     l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
        #     l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
        # l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2,
        #                                                                                        labels[:, 1,
        #
        #
        #                                                                                        ...]) * ce_weight
        l_g_total=0
        labels-=1
        l_g_total+=self.cri_ce_loss(pred1, labels[:, 0, ...])+self.cri_ce_loss(pred2, labels[:, 1, ...])



        return l_g_total

    def  val(self, epoch,model=None,multi_outputs=False):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net
        else:
            val_model=model.netG
            val_model.eval()
        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():

                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if self.config['network_G_CD']['which_model_G'] == 'Feat_Cmp':
                    labels0 = F.interpolate(labels, size=torch.Size(
                        [imgs.shape[2] // self.config.ds, imgs.shape[3] // self.config.ds]), mode='nearest')
                    labels0[labels == 1] = -1  # change
                    labels0[labels == 0] = 1  # must convert ot [-1,1] before calculating  loss

                    featT1, featT2 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    dist = F.pairwise_distance(featT1, featT2, keepdim=True)
                    dist = F.interpolate(dist, size=imgs.shape[2:], mode='bilinear', align_corners=True)

                    outputs = (dist > 1).float()
                    loss = self.cri_dist(dist, labels0)

                elif self.config['network_G_CD']['which_model_G'] == 'FC_EF' or self.config['network_G_CD']['which_model_G'] == 'Seg_EF':
                    outputs = val_model(imgs)
                    bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                    loss = bce_loss(outputs, labels)
                else:
                    _,_,_,outputs=val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    ce_loss=nn.CrossEntropyLoss()
                    loss=ce_loss(outputs,labels)


                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc+=f1_score

        val_loss=lossAcc*1.0/(len(self.valDataloader))
        val_acc=correctsAcc*1.0/(len(self.valDataloader))

        print('Epoch %d evaluate done ' % epoch)
        # convert to train mode for next training
        if model==None:
            self.net.train()

        del outputs
        torch.cuda.empty_cache()

        return  val_loss,val_acc


