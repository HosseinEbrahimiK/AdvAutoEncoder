import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class PaddingLayer(nn.Module):

    def __init__(self, ksize, stride):
        super(PaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)

        x = F.pad(x, padding, "constant", 0)

        return x

class ResBlock(nn.Module):

  def __init__(self, in_chans, nr_featmaps, nr_units, stride=1):
        super(ResBlock, self).__init__()
        
        if in_chans != nr_featmaps[-1] or stride != 1:
          self.shortcut_conv = nn.Conv2d(in_chans, nr_featmaps[-1], kernel_size=1, stride=stride)
        
        else:
          self.shortcut_conv = None
        
        self.inp_ch = in_chans
        self.feat_maps = nr_featmaps
        self.cnt_units = nr_units
        
        self.units = nn.ModuleList()

        unit_in_chan = in_chans

        for i in range(nr_units):

          unit_layer = [
                        
          ## conv 1                  
          ("bn1", nn.BatchNorm2d(unit_in_chan, eps=1e-5)),
          ("relu1", nn.ReLU(inplace=True)),
          ("conv1", nn.Conv2d(unit_in_chan, nr_featmaps[0], kernel_size= 1, stride=1, bias=False)),

          ## conv 2
          ("bn2", nn.BatchNorm2d(nr_featmaps[0], eps=1e-5)),
          ("relu2", nn.ReLU(inplace=True)),
          ("pad", PaddingLayer(3, stride=stride if i==0 else 1)), 
          ("conv2", nn.Conv2d(nr_featmaps[0], nr_featmaps[1], kernel_size=3, stride=stride if i==0 else 1, bias=False)),

          ## conv 3
          ("bn3", nn.BatchNorm2d(nr_featmaps[1], eps=1e-5)),
          ("relu3", nn.ReLU(inplace=True)),
          ("conv3", nn.Conv2d(nr_featmaps[1], nr_featmaps[2], kernel_size=1, stride=1, bias=False))
          ]

          if i == 0:
            unit_layer = unit_layer[2:]

          self.units.append(nn.Sequential(OrderedDict(unit_layer)))
          unit_in_chan = nr_featmaps[-1]

        self.blk_bna = nn.Sequential(
          OrderedDict(
              [
                  ("bn", nn.BatchNorm2d(nr_featmaps[-1], eps=1e-5)),
                  ("relu", nn.ReLU(inplace=True)),
              ]
          )
        )

  def forward(self, x):

    if self.shortcut_conv is None:
      shortcut = x
    else:
        shortcut = self.shortcut_conv(x)
    
    for i in range(len(self.units)):
      next_x = self.units[i](x) + shortcut
      x = next_x
      shortcut = next_x
    
    out = self.blk_bna(next_x)

    return out

class DenseBlock(nn.Module):

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super(DenseBlock, self).__init__()

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        unit_in_ch = in_ch
        self.units = nn.ModuleList()

        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("bn0", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                            ("relu0", nn.ReLU(inplace=True)),
                            ("conv1", nn.Conv2d(unit_in_ch, unit_ch[0], unit_ksize[0], stride=1, padding=0, bias=False)),
                         
                            ("bn1", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                            ("relu1", nn.ReLU(inplace=True)),
                            ("conv2", nn.Conv2d(unit_ch[0], unit_ch[1], unit_ksize[1], groups=split, stride=1, padding=0, bias=False))
                        ]
                    )
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = crop_to_shape(prev_feat, new_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat

class UpSample2x(nn.Module):

    def __init__(self):
        super(UpSample2x, self).__init__()
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        x = x.unsqueeze(-1)
        mat = self.unpool_mat.unsqueeze(0)
        ret = torch.tensordot(x, mat, dims=1)
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image