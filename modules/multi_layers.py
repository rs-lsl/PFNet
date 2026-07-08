import os
import sys
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
# from timm.models.layers import DropPath, trunc_normal_
import math
import numpy as np


class SelfAttention_patch(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32, patch_size=4):
        super().__init__()

        self.n_head = n_head
        self.patch_size = patch_size

        self.norm = nn.GroupNorm(norm_groups, in_channel)  # each group will have 4 channels
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm)#.view(batch, n_head, head_dim * 3, height, width)
        # print(qkv.shape)
        # print(qkv.unfold(2, self.patch_size, self.patch_size).shape)
        # print(qkv.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).shape)
        qkv = qkv.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)\
            .permute(0,1,4,5,2,3).reshape(batch, n_head, head_dim*3, self.patch_size*self.patch_size, -1).permute(0,1,3,2,4)
        # print(qkv.shape)

        query, key, value = qkv.chunk(3, dim=3)  # bhdyx
        # print(query.size())
        height_p = height//self.patch_size  # 16
        width_p = width//self.patch_size   # 8
        # query = query.view(batch, n_head, head_dim, height_p, self.patch_size,
        #                    width_p, self.patch_size).permute(0,1,4,6,2,3,5)
        # # print(query.shape)
        # key = key.view(batch, n_head, head_dim, height_p, self.patch_size,
        #                    width_p, self.patch_size).permute(0,1,4,6,2,3,5)
        # value = value.view(batch, n_head, head_dim, height_p, self.patch_size,
        #                    width_p, self.patch_size).permute(0,1,4,6,2,3,5)
        attn = torch.einsum(
            "bnpch, bnpcw -> bnphw", query, key
        ).contiguous() / math.sqrt(channel)
        # attn = attn.view(batch, n_head, self.patch_size, self.patch_size, height_p, width_p, -1)
        attn = torch.softmax(attn, -1)
        # attn = attn.view(batch, n_head, self.patch_size, self.patch_size, height_p, width_p, height_p, width_p)
        # print(attn.shape)
        # print(value.shape)
        out = torch.einsum("bnphw, bnpcw -> bnpch", attn, value).contiguous()
        out = self.out(out.permute(0,1,3,2,4).reshape(batch, channel, self.patch_size, self.patch_size, height_p, width_p)
                       .permute(0,1,2,4,3,5).reshape(batch, channel, height, width))

        return out + input