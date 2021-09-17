from configs.config_utils import process_config, get_train_args
import numpy as np
from data_loaders.data_proc import RandomCropWeight7,RandomFlip,RandomRotate,RandomShiftScaleRotate,RandomHueSaturationValue
from data_loaders.data_proc import RandomShiftScale,RandomTranspose,RandomNoise,RandomColor,RandomColor2,RandomMix,RandomCutMix,RandomScale,RandomMixorScale,RandomCropResizeWeight7
from trainers.trainer_optim_CD import TrainerOptim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os
from sklearn.model_selection import train_test_split
from data_loaders.data_proc import ImagesetDatasetCD,ImagesetDatasetCD_Sense,ToTensor_Sense
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):#seem not to work

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
# fix random seed
seed=37148
rng = np.random.RandomState(seed)#
# torch.manual_seed(seed)##为CPU设置随机种子
# #torch.cuda.manual_seed()#为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed)##设置用于在所有GPU上生成随机数的种子。 如果CUDA不可用，可以安全地调用此函数；在这种情况下，它将被静默地忽略。为所有GPU设置随机种子can make the result the same, but the acc is lower





def main_train():
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
    #may worse the performance when torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.deterministic = True# 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法 for REPRODUCIBILITY https://www.zhihu.com/question/325482866
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True#使用benchmark以启动CUDNN_FIND自动寻找最快的操作，当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能

    #==========================using new dataloader for building and road extraction====================================
    data_dir= config.data_dir
    val_proportion = config.val_proportion
    batch_size = config.batch_size
    num_worker = config.num_worker
    # #label_type='label32'
    # label_type = 'label255'#for BCD
    # #label_type = 'label7'
    use_score = True
    if config.dataset_name == "sensetime" or config.dataset_name == "HRSCD":
        train_set_src = [pic for pic in os.listdir(os.path.join(data_dir, 'train', 'im1'))]
        train_list_src, val_list_src = train_test_split(train_set_src,
                                                        test_size=val_proportion,
                                                        random_state=1, shuffle=True)

        message1 = "the number of train and val data for source dataset is {:6d} and {:6d}".format(len(train_list_src),
                                                                                                   len(val_list_src))
        train_transforms = transforms.Compose([

            RandomFlip(),
            RandomRotate(),
            RandomTranspose(use_CD=True),
            #RandomMixorScale(),#only for MC7

            ToTensor_Sense(use_rgb=True)#for MC7
            #ToTensor_Sense(use_label255=True)  # for SC2
            # ToTensor_Sense(use_label32=True)#for MC32
        ])

        test_transforms = transforms.Compose([

            ToTensor_Sense(use_rgb=True)#for MC7
            #ToTensor_Sense(use_label255=True)  # for SC2
            # ToTensor_Sense(use_label32=True)#for MC32
        ])
        label_type = 'label7'
        train_dataset_src = ImagesetDatasetCD_Sense(imset_list=train_list_src, config=config, label_type=label_type,
                                                    mode='Train',
                                                    transform=train_transforms)#for SC2

        #val_list_src = [pic for pic in os.listdir(os.path.join(data_dir, 'test', 'im1'))]
        val_dataset_src = ImagesetDatasetCD_Sense(imset_list=val_list_src, config=config, label_type=label_type, use_score=use_score,
                                                mode='Val',
                                                transform=test_transforms)


















    train_dataloader_src = DataLoader(train_dataset_src, batch_size=batch_size,
                                  shuffle=True, num_workers=num_worker,
                                  # collate_fn=collateFunction(),  # 如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
                                  pin_memory=True,drop_last=True)  # len(train_dataloader)

    val_dataloader_src = DataLoader(val_dataset_src, batch_size=batch_size*2,
                                    # can be set a large value due to torch.no_grad
                                    shuffle=True, num_workers=num_worker,
                                    pin_memory=True, drop_last=True)  # still len(val_dataloader), for if one-single image is left the acc will not be accurate





    print('[INFO] 构造网络...')

    print('[INFO] 训练网络...')

    #==========================for traning using optim of each iteration==========================
    trainer = TrainerOptim(config, train_dataloader_src, val_dataloader_src)
    if config["train"]["fine_tune"]:

        trainer.train_optim_tune()
    else:
        trainer.train_optim()#define different optim_function for the same netG, otherwise define differnt model.py for different task, such as CD,SR

    print('[INFO] 训练完成...')

if __name__ == '__main__':
    main_train()