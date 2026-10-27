"""
Note: 1. Given the projected [px, py], we use bilinear interpolation to collection the image feature representation
Author: Yuhang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImgFeatAggregator(nn.Module):
    def __init__(self, interp_method = 'bilinear'):
        super(ImgFeatAggregator, self).__init__()
        assert interp_method in ['bilinear', 'bicubic', 'nearest']
        self.inter_method = interp_method

    def interp_imgfeat(self, imgfeat, grid_px_py):
        '''
        Given an imgfeat, we use bilinear sampling to img feat centres around [px, py]
        TODO: double check [px, py], which direction is along the height and which direction is along the width
        :param imgfeat: [N, C, Hin, Win]
        :param grid_px_py: [N, Hout, Wout, 2]
        :return: [N, C, Hout, Wout],
        '''
        sampled_grid_feat = F.grid_sample(input=imgfeat,
                                          grid=grid_px_py,
                                          mode=self.inter_method,
                                          padding_mode='zeros')

        return sampled_grid_feat
    def forward(self, imgfeat, grid_px_py):

        return self.interp_imgfeat(imgfeat, grid_px_py)