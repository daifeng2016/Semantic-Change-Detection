from configs.config_utils import process_config, get_train_args
import  os
import numpy as np
from data_loaders.data_proc import ImagesetDatasetCD,ReSize,RandomCrop,ImagesetDatasetCD_Sense
from infers.infers import Infer
from torch.utils.data import Dataset, DataLoader
from data_loaders.data_proc import ToTensor_Sense
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from models.utils import print_model_parm_nums
import torch
import models.Satt_CD.networks as networks
from utils.ptflops2 import get_model_complexity_info
# fix random seed
rng = np.random.RandomState(37148)

def input_constructor(input_shape):
    batch = torch.ones(()).new_empty((1, *input_shape)).cuda()
    return batch,batch

def print_model_para(save_path,net):
    import sys
    mylog = open(save_path, 'w')
    stdout_backup = sys.stdout
    sys.stdout = mylog


    macs, params = get_model_complexity_info(net, (3, 512, 512),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             ost=mylog)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    mylog.close()
    sys.stdout = stdout_backup
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def main_test():
    print('[INFO] 解析配置...')
    parser = None
    config = None
    try:
        args, parser = get_train_args()
        config = process_config(args.config)  # json 文件中分隔符 必须是 /
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        if parser:
            parser.print_help()
        print('[Exception] 参考: python main_train.py -c configs/config.json')
        exit(0)

    print('[INFO] 加载数据...')
    config.mode='Test'
    #===========================for  test===================================
    # test_transforms = transforms.Compose([
    #
    #    #ToTensor_Sense(use_rgb=True)#for MC7
    #    ToTensor_Sense(use_label255=True)#for SC2
    #    #ToTensor_Sense(use_label32=True)#for MC32
    # ])

    #=============================use imagedataset==============================
    use_score=True#false for whole test
    config.use_CRF = False
    # ====for other_dir test==================
    #data_dir = config.data_dir00
    #===========for current_dir test==========
    data_dir = config.data_dir
    #=========================================
    #label_type = 'label32'
    #label_type = 'label255'
    #label_type = 'label7'

    # test_set = [pic for pic in os.listdir(os.path.join(data_dir, 'test', 'im1'))]  # for sensetime dataset
    # test_dataset = ImagesetDatasetCD_Sense(imset_list=test_set, config=config,use_score=use_score,label_type=label_type,
    #                                        mode='Test', transform=test_transforms)#for MC7

    if config.dataset_name=="sensetime" or config.dataset_name=="HRSCD":
        #test_set = [pic for pic in os.listdir(os.path.join(data_dir, 'test', 'im1'))]
        test_transforms = transforms.Compose([

             ToTensor_Sense(use_rgb=True)#for MC7
            #ToTensor_Sense(use_label255=True)  # for SC2
            # ToTensor_Sense(use_label32=True)#for MC32
        ])
        label_type = 'label7'
        test_set = [pic for pic in os.listdir(os.path.join(data_dir, 'test', 'im1'))]  # for sensetime dataset
        test_dataset = ImagesetDatasetCD_Sense(imset_list=test_set, config=config, use_score=use_score,
                                             label_type=label_type,
                                             mode='Test', transform=test_transforms)






   #for whu-cd

    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                  shuffle=False, num_workers=config.num_worker,
                                pin_memory=True
                                  )


    print('[INFO] 预测网络...')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.mode = 'Test'
    #config.use_CRF=False
    if config.use_CRF:#######
        print("prediction using CRF...")
    config["network_G_CD"]["training_mode"]=False
    model = networks.define_G_CD(config).to(device)
    #======================model params=====================================
    # save_path = os.path.join(config.model_dir, config.pred_name + '_param_complexity.txt')
    # print_model_para(save_path, model)
    # return 0
    #=======================================================================
    print('[INFO] 预测数据...')
    infer=Infer(config,model,test_dataloader)
    if config.model=='MRCD':
          infer.CD_Evaluation_Sense(use_score=use_score,use_TTA=True,use_model_ensemble=False,output_rgb=True,use_con=False,use_CRF=config.use_CRF,mode='_final_iter')
    else:
          f1_score_value, acc, kappa=infer.CD_Evaluation(use_TTA=True)#for SC2 use best_acc as default
          print("f1_score is %.4f" % f1_score_value)
          print("overall accuracy is %.4f" % acc)
          print("kappa is %.4f" % kappa)



    print('[INFO] 预测完成...')


if __name__ == '__main__':
    main_test()