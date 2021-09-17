import torch
import torch.nn.functional as F
from torch import nn


class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # [1,512,8] B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # [1,8,512] B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # [1,512,512] transpose check
        energy = (self.key_channel**-.5) * energy#[1,512,512]

        attention = self.softmax(energy)  # [1,512,512]BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  #  [1,64,512] B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))#[1,64,512]
        out = out.view(m_batchsize, C, width, height)#[1,64,16,32]

        out = F.interpolate(out, [width*self.ds,height*self.ds])#[1,64,16,32]
        out = out + input

        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))  # 不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.softmax = nn.Softmax(
            dim=-1)  # 对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1 ,dim:指明维度，dim=0表示按列计算；dim=1表示按行计算。默认dim的方法已经弃用了

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()  # [1,3,128,128]
        proj_query = x.view(m_batchsize, C, -1)  # [1,3,16384]
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # [1,16384,3]

        energy = torch.bmm(proj_query, proj_key)  # [1,3,16384].[1,16384,3]=[1,3,3]
        # torch.max()[0]， 只返回最大值的每个数  troch.max()[1]， 只返回最大值的每个索引

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy  # [1,3,3]
        attention = self.softmax(energy_new)  # [1,3,3]
        proj_value = x.view(m_batchsize, C, -1)  ##[1,3,16384]

        out = torch.bmm(attention,
                        proj_value)  # [1,3,3].[1,3,16384]=[1,3,16384]  == weights = torch.matmul(affinity_new, proj_value)
        out = out.view(m_batchsize, C, height, width)  # [1,3,16384]==>[1,64,128,128]

        out = self.gamma * out + x
        return out


class PCAM(nn.Module):
    def __init__(self, in_dim):
        super(PCAM, self).__init__()
        self.chanel_in = in_dim
        self.PAM=BAM(in_dim)
        self.CAM=CAM_Module(in_dim)
        self.conv=nn.Conv2d(in_dim*2,in_dim,3,1,1)
    def forward(self,x):

        out1=self.PAM(x)
        out2=self.CAM(x)
        return torch.cat([out1,out2],dim=1)






class _PAMBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input/Output:
        N * C  *  H  *  (2*W)
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to partition the input feature maps
        ds                : downsampling scale
    '''
    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)



    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)
        # input shape: b,c,h,2w
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)//2


        local_y = []
        local_x = []
        step_h, step_w = h//self.scale, w//self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i*step_h, j*step_w
                end_x, end_y = min(start_x+step_h, h), min(start_y+step_w, w)
                if i == (self.scale-1):
                    end_x = h
                if j == (self.scale-1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:,:,:,w:]], 4)  # B*N*H*W*2
        query = torch.stack([query[:, :, :, :w], query[:,:,:,w:]], 4)  # B*N*H*W*2
        key = torch.stack([key[:, :, :, :w], key[:,:,:,w:]], 4)  # B*N*H*W*2


        local_block_cnt = 2*self.scale*self.scale

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)

            query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)

            sim_map = torch.bmm(query_local, key_local)  # batch matrix multiplication
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(value_local, sim_map.permute(0,2,1))
            # context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size_new, self.value_channels, h_local, w_local, 2)
            return context_local

        #  Parallel Computing to speed up
        #  reshape value_local, q, k
        v_list = [value[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        v_locals = torch.cat(v_list,dim=0)
        q_list = [query[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        q_locals = torch.cat(q_list,dim=0)
        k_list = [key[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        k_locals = torch.cat(k_list,dim=0)
        # print(v_locals.shape)
        context_locals = func(v_locals,q_locals,k_locals)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size*(j+i*self.scale)
                right = batch_size*(j+i*self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = torch.cat([context[:,:,:,:,0],context[:,:,:,:,1]],3)


        if self.ds !=1:
            context = F.interpolate(context, [h*self.ds, 2*w*self.ds])

        return context


class PAMBlock(_PAMBlock):
    def __init__(self, in_channels, key_channels=None, value_channels=None, scale=1, ds=1):
        if key_channels == None:
            key_channels = in_channels//8
        if value_channels == None:
            value_channels = in_channels
        super(PAMBlock, self).__init__(in_channels,key_channels,value_channels,scale,ds)


class PAM(nn.Module):
    """
        PAM module
    """

    def __init__(self, in_channels, out_channels, sizes=([1]), ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds  # output stride
        self.value_channels = out_channels
        self.key_channels = out_channels // 8


        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, self.key_channels, self.value_channels, size, self.ds)
             for size in sizes])
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels * self.group, out_channels, kernel_size=1, padding=0,bias=False),
            # nn.BatchNorm2d(out_channels),
        )

    def _make_stage(self, in_channels, key_channels, value_channels, size, ds):
        return PAMBlock(in_channels,key_channels,value_channels,size,ds)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]

        #  concat
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn(torch.cat(context, 1))

        return output
