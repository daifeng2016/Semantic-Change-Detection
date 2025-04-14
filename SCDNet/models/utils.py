import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.networks_other import init_weights
import numpy as np
from PIL import Image
import copy
import random
from torch.autograd import Variable

from torch.nn import init
import functools

from torch.optim import lr_scheduler
import time



###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


'''
Xavier在tanh中表现的很好，但在Relu激活函数中表现的很差，所何凯明提出了针对于Relu的初始化方法。
Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification He, K. et al. (2015)
该方法基于He initialization,其简单的思想是：
在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0，所以，要保持方差不变，只需要在 Xavier 的基础上再除以2
https://www.cnblogs.com/jfdwd/p/11078372.html
使用fan_in保持weights的方差在前向传播中不变；使用fan_out保持weights的方差在反向传播中不变
'''


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in',
                             nonlinearity='relu')  # default nonlinearity='leaky_relu'    for gan
        # init.kaiming_normal_(m.weight.data, mode='fan_out',nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')  # for gan
        # init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in',
                             nonlinearity='relu')  # default nonlinearity='leaky_relu'    for gan
        # init.kaiming_normal_(m.weight.data, mode='fan_out',nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')  # for gan
        # init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to a fixed number"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_scheduler(optimizer, opt):
    print('opt.lr_policy = [{}]'.format(opt.lr_policy))
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'step2':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.01, patience=5)
    elif opt.lr_policy == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'step_warmstart':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step_warmstart2':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 50:
                lr_l = 1
            elif 50 <= epoch < 100:
                lr_l = 0.1
            elif 100 <= epoch:
                lr_l = 0.01
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_n_parameters(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params


def measure_fp_bp_time(model, x, y):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    y_pred = model(x)
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    if isinstance(y_pred, tuple):
        y_pred = sum(y_p.sum() for y_p in y_pred)
    else:
        y_pred = y_pred.sum()

    # zero gradients, synchronize time and measure
    model.zero_grad()
    t0 = time.time()
    # y_pred.backward(y)
    y_pred.backward()
    torch.cuda.synchronize()
    elapsed_bp = time.time() - t0
    return elapsed_fp, elapsed_bp


def benchmark_fp_bp_time(model, x, y, n_trial=1000):
    # transfer the model on GPU
    model.cuda()

    # DRY RUNS
    for i in range(10):
        _, _ = measure_fp_bp_time(model, x, y)

    print('DONE WITH DRY RUNS, NOW BENCHMARKING')

    # START BENCHMARKING
    t_forward = []
    t_backward = []

    print('trial: {}'.format(n_trial))
    for i in range(n_trial):
        t_fp, t_bp = measure_fp_bp_time(model, x, y)
        t_forward.append(t_fp)
        t_backward.append(t_bp)

    # free memory
    del model

    return np.mean(t_forward), np.mean(t_backward)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


#
def entropy_loss_ita(v,ita=2.0):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: #====batch_size x 1 x h x w====# 1

    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    #vv=torch.sum(torch.mul(v, torch.log2(v + 1e-30)),dim=0)#[4,2,256,256]==>[2,256,256]
    #torch.sum(torch.mul(v, torch.log2(v + 1e-30)), dim=1)##[4,2,256,256]==>[4,256,256]
    if c>1:
       constant=np.log2(c)
    else:
       constant=1

    ent= -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * constant)#c!=1!!!
    ent = ent ** 2.0 + 1e-8
    ent = ent ** ita
    loss_ent = ent.mean()

    return loss_ent



def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: #====batch_size x 1 x h x w====# 1
    """
    assert v.dim() == 4
    n, c, h, w = v.size()

    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
        output.size()==input.size()
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def adjust_LR_iter(optimizer, i_iter,learning_rate,num_steps):
    def lr_poly(base_lr, iter, max_iter, power=0.9):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))
    lr = lr_poly(learning_rate, i_iter, num_steps)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10


def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1, 2, 0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output


def find_good_maps(D_outs, pred_all,thresh_ST=0.6,batch_size=8):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > thresh_ST:
            count += 1

    if count > 0:
        print('Above ST-Threshold : ', count, '/', batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0
        for j in range(D_outs.size(0)):
            if D_outs[j] > thresh_ST:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel += 1
        return pred_sel.cuda(), label_sel.cuda(), count
    else:
        return 0, 0, count


def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)#[4,321,321]==>[4,1,321,321]
    D_label = np.ones(ignore_mask.shape)*label#[4,1,321,321]
    D_label[ignore_mask] = 255# if ignore_mask=false, then all elements of D_label is 1
    D_label = Variable(torch.FloatTensor(D_label)).cuda(0)##[4,1,321,321]

    return D_label

def find_good_regions(D_outs, pred_all,thresh_ST=0.6):
    '''
    generate high-confidence region based on D value for semi-supervised learning

    :param D_outs:
    :param pred_all:
    :param thresh_ST:
    :param batch_size:
    :return:
    '''
    semi_ignore_mask=D_outs.data.cpu().numpy().squeeze(axis=1)<thresh_ST
    #semi_ignore_count=D_outs.numel()#for torch tensor
    semi_gt=pred_all.data.cpu().numpy().argmax(axis=1)#generate the semi_gt based on the argmax of the pred tensor
    semi_gt[semi_ignore_mask]=255
    semi_ratio=1.0 - float(semi_ignore_mask.sum()*1.0/semi_ignore_mask.size)#semi_ignore_mask.size=size prod of ndarray(semi_ignore_mask)
    print('semi ratio: {:.4f}'.format(semi_ratio))
    if semi_ratio>0:
        return  semi_ratio, torch.FloatTensor(semi_gt).cuda(0)
    else:
        return semi_ratio,0


def find_good_regions2(pred_all,thresh_ST=0.6):
    '''
    generate high-confidence region based on prediction value for semi-supervised learning
    # can not generate labels by uisng pred, otherwise will cause self-referee problem
    :param D_outs:
    :param pred_all:
    :param thresh_ST:
    :param batch_size:
    :return:
    '''
    soft_label= F.softmax(pred_all,dim=1).detach()  # otherwise one of the variables needed for gradient computation has been modified by an inplace operation

    soft_label=soft_label[:, 1].unsqueeze(1)# can not generate labels by uisng pred, otherwise will cause self-referee problem
    semi_ignore_mask=soft_label.data.cpu().numpy().squeeze(axis=1)<thresh_ST#[16,128,128]
    #semi_ignore_count=D_outs.numel()#for torch tensor
    semi_gt=pred_all.data.cpu().numpy().argmax(axis=1)#[16,128,128] generate the semi_gt based on the argmax of the pred tensor
    semi_gt[semi_ignore_mask]=255#ignore value is 255
    semi_ratio=1.0 - float(semi_ignore_mask.sum()*1.0/semi_ignore_mask.size)#semi_ignore_mask.size=size prod of ndarray(semi_ignore_mask)

    if semi_ratio>0:
        print('semi ratio: {:.4f}'.format(semi_ratio))
        return  semi_ratio, torch.FloatTensor(semi_gt).cuda(0)
    else:
        return semi_ratio,0








def calpsnr(im1_tensor, im2_tensor,max255=False):
    '''
    pnsr:https://blog.csdn.net/u010886794/article/details/84784453
    :param im1_tensor:
    :param im2_tensor:
    :return:
    '''
    im1=im1_tensor.data.cpu().numpy()
    im2 = im2_tensor.data.cpu().numpy()
    if max255==False:
       im1=im1*255
       im2=im2*255
    diff = np.abs(im1 - im2)
    rmse = np.sqrt(np.mean(np.square(diff)))
    psnr = 20*np.log10(255/rmse)
    return psnr

def DeNormalization(rgb_mean,im_tensor):
    im=im_tensor.data.cpu().numpy()[0].transpose([1,2,0])
    im=(im+rgb_mean)*255
    im=im.astype('uint8')
    return Image.fromarray(im,'RGB')

def DeNormalization2(rgb_mean,im_tensor):
    im=im_tensor.data.cpu().numpy()[0].transpose([1,2,0])
    #im=(im+rgb_mean)*255
    im=im.astype('uint8')
    return Image.fromarray(im,'RGB')


def Acc(mask_probs,true_masks):
    masks_probs_flat = mask_probs.view(-1)
    true_masks_flat = true_masks.view(-1)
    masks_probs_flat = masks_probs_flat > 0.5
    correctsAcc = torch.sum(
        masks_probs_flat.int() == true_masks_flat.int()).item() * 1.0 / true_masks_flat.size(0)
    return correctsAcc

def IOU(mask_probs,true_masks):
    smooth = 1.0  # may change
    mask_probs=(mask_probs>0.6).float()
    i = torch.sum(true_masks)  # 对二维矩阵的所有元素求和
    j = torch.sum(mask_probs)
    intersection = torch.sum(true_masks * mask_probs)
    #score = (2. * intersection + smooth) / (i + j + smooth)#bce
    score = (intersection + smooth) / (i + j - intersection + smooth)#iou
    return score.mean()

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return mean_iu, cls_iu

def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad
def one_hot(label,num_classes=2):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)


def one_hot_raw(label,num_classes=7):
    #label = label.numpy()
    one_hot = np.zeros((label.shape[0],label.shape[1],  num_classes), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,:,i] = (label==i)
    #handle ignore labels
    return one_hot




def one_hot_cuda(label,num_classes=2,label_smooth=False):
    label = label.data.cpu().numpy()
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    if label_smooth:
        #smooth_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / num_classes
        smooth_value=0.1
        one_hot=(1.0-smooth_value)*one_hot+smooth_value/num_classes

    return torch.FloatTensor(one_hot).cuda()




def make_one_hot(labels, num_class=2, gpu_id=0):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    C=num_class
    labels = labels.long()
    try:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        one_hot = one_hot.cuda(0)
    except:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target

# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

'''
To be used to apply gaussian noise in the input to the discriminator
'''

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))# transform list to tensor?

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = torch.zeros(x.size()).normal_() * scale
            x = x + sampled_noise
        return x

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=1, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        #====initialise the blocks=====
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class FCNConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(FCNConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class UnetGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetGatingSignal3, self).__init__()
        self.fmap_size = (4, 4, 4)

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, (1,1,1), (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(in_size//2),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool3d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size//2) * self.fmap_size[0] * self.fmap_size[1] * self.fmap_size[2],
                                 out_features=out_size, bias=True)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, (1,1,1), (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool3d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size//2) * self.fmap_size[0] * self.fmap_size[1] * self.fmap_size[2],
                                 out_features=out_size, bias=True)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        batch_size = inputs.size(0)
        outputs = self.conv1(inputs)
        outputs = outputs.view(batch_size, -1)
        outputs = self.fc1(outputs)
        return outputs

class UnetGridGatingSignal2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1), is_batchnorm=True):
        super(UnetGridGatingSignal2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs



class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class Attention_block(nn.Module):#addictive attention
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
class unetUp_Att(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp_Att, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')
        self.Att=Attention_block(F_g=out_size,F_l=out_size,F_int=int(out_size/2))
    def forward(self, inputs1, inputs2):#[1,128,32,32]+[1,256,16,16]+conv==>[1,128,32,32]
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        outputs1_att=self.Att(g=outputs2,x=outputs1)


        return self.conv(torch.cat([outputs1_att, outputs2], 1))

class unetConv2_res(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=1, ks=3, stride=1, padding=1, use_res=True,
                 use_att=False,act='relu'):
        super(unetConv2_res, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.use_res = use_res
        if act=='relu':
            activation=nn.ReLU(inplace=True)
        else:
            activation=nn.LeakyReLU(0.2,inplace=True)

        if is_batchnorm:
            self.conv0 = conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                nn.Conv2d(in_size, out_size, ks, s, p),
                nn.BatchNorm2d(out_size),
                #nn.ReLU(inplace=False)
                activation
            )
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                            nn.BatchNorm2d(out_size),
                                            )
            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    #nn.ReLU(inplace=False)
                    activation
                )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size
            self.conv_last=nn.Sequential(
                nn.Conv2d(out_size, out_size, ks, s, p),
                nn.BatchNorm2d(out_size)
            )
        else:
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
            self.conv0 = conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, s, p),
                #nn.ReLU(inplace=False)
                activation
            )

            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    #nn.ReLU(inplace=False)
                    activation
                )
                setattr(self, 'conv%d' % i, conv)
            self.conv_last = nn.Conv2d(out_size, out_size, ks, s, p)#not use act for the last conv
        # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        x = self.conv0(x)
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)  # [9,16,256,256]
        x_short = self.conv_short(inputs)
        x=self.conv_last(x)
        return x + x_short

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)



class unetConv2_res_IN(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=1, ks=3, stride=1, padding=1, use_res=True,
                 use_att=False,act='relu'):
        super(unetConv2_res_IN, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.use_res = use_res
        if act=='relu':
            activation=nn.ReLU(inplace=True)
        else:
            activation=nn.LeakyReLU(0.2,inplace=True)

        if is_batchnorm:
            self.conv0 = conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                nn.ReflectionPad2d(p),
                nn.Conv2d(in_size, out_size, ks, s),
                nn.InstanceNorm2d(out_size),
                activation

            )
            self.conv_short = nn.Sequential(
                nn.ReflectionPad2d(p),
                nn.Conv2d(in_size, out_size, ks, s),
                                            nn.BatchNorm2d(out_size),
                                            )
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.ReflectionPad2d(p),
                    nn.Conv2d(out_size, out_size, ks, s),
                    nn.InstanceNorm2d(out_size),
                    activation

                )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size
            self.conv_last=nn.Sequential(
                nn.ReflectionPad2d(p),
                nn.Conv2d(out_size, out_size, ks, s),
                nn.InstanceNorm2d(out_size),
                activation
            )
        else:
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
            self.conv0 = conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, s, p),
                #nn.ReLU(inplace=False)
                activation
            )

            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    #nn.ReLU(inplace=False)
                    activation
                )
                setattr(self, 'conv%d' % i, conv)
            self.conv_last = nn.Conv2d(out_size, out_size, ks, s, p)


    def forward(self, inputs):
        x = inputs
        x = self.conv0(x)
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)  # [9,16,256,256]
        x_short = self.conv_short(inputs)
        x=self.conv_last(x)
        return x + x_short







class unetConv2_res0(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=1, ks=3, stride=1, padding=1, use_res=True,
                 use_att=False,act='relu'):
        super(unetConv2_res0, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.use_res = use_res
        # self.use_att = use_att
        # self.channel_Att = ChannelAttention(out_size)
        # self.spatial_Att = SpatialAttention()
        # self.SC_Att = SCAttention(out_size)
        if is_batchnorm:
            self.conv0 = conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                nn.Conv2d(in_size, out_size, ks, s, p),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=False)

            )
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                            nn.BatchNorm2d(out_size),
                                            )
            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=False)

                )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size
            self.conv_last=nn.Sequential(
                nn.Conv2d(out_size, out_size, ks, s, p),
                nn.BatchNorm2d(out_size)
            )
        else:
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
            self.conv0 = conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, s, p),
                nn.ReLU(inplace=False)
            )

            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.ReLU(inplace=False)
                )
                setattr(self, 'conv%d' % i, conv)
            self.conv_last = nn.Conv2d(out_size, out_size, ks, s, p)
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        x = self.conv0(x)
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)  # [9,16,256,256]
        x_short = self.conv_short(inputs)
        x=self.conv_last(x)
        return x + x_short

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv,use_res=False,act='relu'):
        super(unetUp, self).__init__()
        if use_res:
            self.conv=unetConv2_res(in_size, out_size, is_batchnorm=True,act=act)
        else:
            self.conv = unetConv2(in_size, out_size, is_batchnorm=True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):#[1,128,16,16,5]+[1,256,8,8,5]
        outputs2 = self.up(inputs2)#[1,256,8,8,5]==>[1,256,16,16,5]
        offset = outputs2.size()[2] - inputs1.size()[2]#0
        padding = 2 * [offset // 2, offset // 2, 0]#[0,0,0,0,0,0]
        outputs1 = F.pad(inputs1, padding)#[1,128,16,16,5]==>[1,128,16,16,5]
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


# Squeeze-and-Excitation Network
class SqEx(nn.Module):

    def __init__(self, n_features, reduction=6):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool3d(x, kernel_size=x.size()[2:5])
        y = y.permute(0, 2, 3, 4, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 4, 1, 2, 3)
        y = x * y
        return y

class UnetUp3_SqEx(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm):
        super(UnetUp3_SqEx, self).__init__()
        if is_deconv:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        concat = torch.cat([outputs1, outputs2], 1)
        gated  = self.sqex(concat)
        return self.conv(gated)

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock, self).__init__()

        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class residualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBottleneck, self).__init__()
        self.convbn1 = nn.Conv2DBatchNorm(in_channels,  n_filters, k_size=1, bias=False)
        self.convbn2 = nn.Conv2DBatchNorm(n_filters,  n_filters, k_size=3, padding=1, stride=stride, bias=False)
        self.convbn3 = nn.Conv2DBatchNorm(n_filters,  n_filters * 4, k_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class SeqModelFeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(SeqModelFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)


def print_model_parm_nums(model):            #得到模型参数总量

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))     #每一百万为一个单位
    return total/1e6

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

### initalize the module
from torch.nn import init
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)