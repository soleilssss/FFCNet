from pickle import TRUE
import torch
from torch.nn.functional import relu, max_pool2d, dropout, dropout2d, adaptive_avg_pool2d
import torch.nn as nn

def complex_relu(input_r,input_i, inplace=True):   # 所以看起来都是自己分了实部和虚部的，然后单独对里面每个进行操作，还是基本使用框架里自带的函数
    return relu(input_r, inplace), relu(input_i, inplace)

def complex_max_pool2d(input_r,input_i,kernel_size = 2, stride=2, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):

    return max_pool2d(input_r, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices), \
           max_pool2d(input_i, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices)

def complex_AdaptiveAvgPool2d(input_r,input_i,output_size=[1,1]):

    return adaptive_avg_pool2d(input_r, output_size), \
           adaptive_avg_pool2d(input_i, output_size)

def complex_cat(combine1_r, combine1_i, combine2_r, combine2_i):   # 实部和虚部分别进行拼接
    out_r = torch.cat([combine1_r, combine2_r], 1)
    out_i = torch.cat([combine1_i, combine2_i], 1)
    return out_r, out_i

def complex_up(img_r, img_i):
    up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    out_r = up(img_r)
    out_i = up(img_i)
    return out_r, out_i

def complex_up_4(img_r, img_i):
    up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    out_r = up(img_r)
    out_i = up(img_i)
    return out_r, out_i

def complex_up_8(img_r, img_i):
    up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    out_r = up(img_r)
    out_i = up(img_i)
    return out_r, out_i

def complex_up_16(img_r, img_i):
    up = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    out_r = up(img_r)
    out_i = up(img_i)
    return out_r, out_i

def complex_dropout(input_r,input_i, p=0.5, training=True, inplace=False):
    return dropout(input_r, p, training, inplace), \
           dropout(input_i, p, training, inplace)


def complex_dropout2d(input_r,input_i, p=0.5, training=True, inplace=False):
    return dropout2d(input_r, p, training, inplace), \
           dropout2d(input_i, p, training, inplace)
