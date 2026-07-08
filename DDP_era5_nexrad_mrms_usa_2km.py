# -*- coding: utf-8 -*-
import os
import sys

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from tqdm import tqdm
import sys
import pandas as pd
import gc
import tempfile
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import pickle
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils_dataset import write_img_gdal
import heapq
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import shutil
import logging
from utils_dataset import create_loader, read_img
import matplotlib.colors as mcolors
import numpy as np


class Pred_model(nn.Module):
    def __init__(self, model, optimizer, dataloader_train, sampler_train, dataloader_val, dataloader_test, const_data, file_name_dem,
                 in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True,
                 time_emb_num=10, results_dir='', device=None, rank=0,
                 local_rank=0, loss_type='', cp_dir=None, args=None, **kwargs):
        super(Pred_model, self).__init__()
        self.args = args
        self.results_dir = results_dir
        self.cp_dir = cp_dir
        self.device = device
        self.rank = rank
        self.local_rank = local_rank
        # self.loss_weight = loss_weight  #
        B, T, C, H, W = in_shape  # T is input_time_length
        self.shape_val = [H, W]
        # self.bs = B
        self.ch = C
        # self.target_dim = args.target_dim # list(range(args.p_dim, args.other_dim))+[15]  # ****************************

        self.dataloader_train, self.sampler_train, self.dataloader_val, self.dataloader_test = \
            dataloader_train, sampler_train, dataloader_val, dataloader_test
        # self.sampler_train = sampler_train
        self.const_data = None  # const_data.type(torch.float32).to(self.device, non_blocking=True)

        self.model = model
        # self.checkpoint_path = os.path.join(self.cp_dir, "initial_weight.pt")
        # print(args)

        if rank == 0:
            print("Total number of paramerters in networks is {}  ".format(
                sum(x.numel() for x in self.model.parameters())))

        log_path = os.path.join(self.results_dir, 'logs', args.ex_name)
        # self.logwriter = LogWriter(logdir=log_path)

        if not args.test:
            # self.steps_per_epoch = len(dataloader_train)
            self.init_optim(optimizer)

        self.init_lat_weight()
        self.init_max_indices()

        self.rain_list = []
        self.norain_list = []
        # self.init_model()
        # self.adv_loss = AdversarialLoss(discriminator, loss_type=loss_type)
        self.dem_data = read_img(file_name_dem)
        self.dem_data = torch.from_numpy((self.dem_data - np.min(self.dem_data)) / (np.max(self.dem_data) - np.min(self.dem_data))).type(
                torch.float32).to(self.device)[None, ...]
        print('self.dem_data.shape', self.dem_data.shape)

        hr_shape = (args.L1_shape[-2], args.L1_shape[-1])
        self.scale_up0, self.scale_up1 = args.dem_shape[0] / hr_shape[0], args.dem_shape[1] / hr_shape[1]
        self.dem_seg_size = (int(args.tar_L1_shape[-2] * self.scale_up0), int(args.tar_L1_shape[-1] * self.scale_up1))

        self.coords_seg_size = (args.tar_size[0], args.tar_size[1])
        self.init_lat_lon()

        stride_h, stride_w = args.crop_stride_test[0], args.crop_stride_test[1]
        self.test_indices = []
        self.test_indices_dem = []
        for lat_idx in range(0, self.sample_lat_size, stride_h):
            for lon_idx in range(0, self.sample_lon_size, stride_w):
                self.test_indices.append((lat_idx, lon_idx))
                self.test_indices_dem.append(self.lr_to_hr_index(lat_idx, lon_idx))

        self.time0 = ['2017-03-29-06', '2017-10-29-12']

    def lr_to_hr_index(self, hr_i, hr_j):
        """将高分辨率索引转换为低分辨率索引"""
        lr_i = int(hr_i * self.scale_up0)  # lat
        lr_j = int(hr_j * self.scale_up1)  # lon
        return lr_i, lr_j

    def init_lat_lon(self):
        # 边界范围
        lat_range = [25, 49]
        lon_range = [245, 291]
        # 生成等间距点
        lats = np.linspace(lat_range[0], lat_range[1], 1201)
        lons = np.linspace(lon_range[0], lon_range[1], 2301)
        # 归一化
        lats_norm = (lats - lat_range[0]) / (lat_range[1] - lat_range[0])
        lons_norm = (lons - lon_range[0]) / (lon_range[1] - lon_range[0])
        # 使用meshgrid创建网格
        lats_grid, lons_grid = np.meshgrid(lats_norm, lons_norm, indexing='ij')
        # indexing='ij' 使得输出形状为 (1200, 2300)
        # 拼接成最终结果
        self.geo_coords = torch.from_numpy(np.concatenate([lats_grid[None, ...], lons_grid[None, ...]], axis=0)).type(
                torch.float32).to(self.device)
        print(f"geo_coords shape: {self.geo_coords.shape}")  # 输出: (2, 1200, 2300)

    def init_max_indices(self):
        hr_shape = (self.args.L1_shape[-2], self.args.L1_shape[-1])
        self.sample_lat_size = hr_shape[0] - self.args.tar_size[0]
        self.sample_lon_size = hr_shape[1] - self.args.tar_size[1]

        stride_h, stride_w = self.args.crop_stride_test[0], self.args.crop_stride_test[1]

        # 直接计算最后一个滑窗起始索引
        last_lat_idx = (self.sample_lat_size - 1) // stride_h * stride_h
        last_lon_idx = (self.sample_lon_size - 1) // stride_w * stride_w

        self.max_indices = (last_lat_idx, last_lon_idx)
        print('self.max_indices', self.max_indices)

    def init_lat_weight(self):

        self.mean_std_era5 = torch.from_numpy(
            np.load(os.path.join(self.args.little_file_path, 'mean_std_era5_391.npy'))).type(
            torch.float32).to(self.device)
        # self.mean_std_era5 = torch.nan_to_num(self.mean_std_era5)
        self.mean_std_era5[1] = torch.where(self.mean_std_era5[1] == 0, 1.0, self.mean_std_era5[1])
        self.mean_std_era5[0] = torch.where(torch.isnan(self.mean_std_era5[0]), 0.0, self.mean_std_era5[0])
        self.mean_std_era5[1] = torch.where(torch.isnan(self.mean_std_era5[1]), 1.0, self.mean_std_era5[1])

        time_weight = torch.from_numpy(np.ones(10)).reshape([1, 10, 1, 1, 1])
        # if self.rank == 0:
        #     print(f'time weight: {time_weight.squeeze()}')
        self.time_weight = time_weight.type(torch.float32).to(self.device)

    def test(self, mode='val'):

        state_dict = torch.load(
            os.path.join(self.cp_dir, "weight.pth"))
        if self.args.dist:
            try:
                self.model.module.load_state_dict(state_dict)
            except:
                self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        self.evaluate(
            metric_list=['mae', 'rmse'], mode=mode
        )

    def merge_patches_efficient(self, patches, original_shape):
        """
        高效版本的patch拼接
        """
        bs, bs_p, time_step, ch, patch_h, patch_w = patches.shape
        # sample_lat_size, sample_lon_size = self.max_indices[0]+self.args.tar_size[0], self.max_indices[1]+self.args.tar_size[1]
        # print(original_shape)
        # stride_h, stride_w = stride

        # 计算网格尺寸
        # num_lat_slices = (sample_lat_size - seg_size[0]) // stride_h + 1
        # num_lon_slices = (sample_lon_size - seg_size[1]) // stride_w + 1

        # 重塑patches以便批量处理
        patches_reshaped = patches.permute(0, 2, 3, 4, 5, 1)  # bs, bs_p, time_step, ch, patch_h, patch_w

        # 使用fold操作进行拼接（更高效）
        # self.args.tar_size, self.args.crop_stride_test
        full_images = F.fold(
            patches_reshaped.reshape(bs * time_step, -1, bs_p),
            output_size=original_shape,
            kernel_size=self.args.tar_L1_shape[-2:],
            stride=self.args.crop_stride_test,
            padding=0
        ).reshape(bs, time_step, ch, original_shape[0], original_shape[1])
        # print(full_images.shape)
        # print(full_images[0, 0, 0])

        # # 计算重叠权重
        ones_patches = torch.ones_like(patches_reshaped.reshape(bs * time_step, -1, bs_p))
        weight_maps = F.fold(
            ones_patches,
            output_size=original_shape,
            kernel_size=self.args.tar_L1_shape[-2:],
            stride=self.args.crop_stride_test,
            padding=0
        ).reshape(bs, time_step, ch, original_shape[0], original_shape[1])
        # print(weight_maps[0, 0, 0])
        result = full_images / weight_maps
        # print(result[0, 0, 0])

        return result

    def evaluate(self, epoch=None, metric_list=['mae', 'mse', 'rmse', 'ssim'], mode='val'):

        forcast_len = self.args.aft_seq_length_test
        dataloader = self.dataloader_test

        self.model.eval()
        time_list = []
        margin_wid = 2
        from datetime import datetime

        now = datetime.now()
        print(len(dataloader))
        with torch.no_grad():
            for step, (qpe_data_full, L1_data, era5_data, qpe_data, time_data) in enumerate(dataloader):
                if step % 1 == 0:
                    print(step)
                bs, bs_p, _, _, _, _ = qpe_data.shape

                L1_data = L1_data.type(torch.float32).flatten(0,1).to(self.device, non_blocking=True)
                qpe_data_full = qpe_data_full.type(torch.float32).to(self.device, non_blocking=True)
                qpe_data = qpe_data.type(torch.float32).flatten(0, 1).to(self.device, non_blocking=True)
                era5_data = era5_data.flatten(0,1).type(torch.float32).to(self.device, non_blocking=True)

                era5_data = torch.nan_to_num(era5_data, nan=0.0)

                era5_data.sub_(self.mean_std_era5[0][None, None, :, None, None])
                era5_data.div_(self.mean_std_era5[1][None, None, :, None, None])

                qpe_data_full = torch.nan_to_num(qpe_data_full, nan=0.0).clamp_(min=0.0)
                qpe_data_full[qpe_data_full > 300] = 0
                labels = qpe_data_full[:, self.args.in_len_val:(self.args.in_len_val + forcast_len)].clone()

                qpe_data = torch.nan_to_num(qpe_data, nan=0.0).clamp_(min=0.0)
                qpe_data[qpe_data > 300] = 0
                qpe_data = (qpe_data - self.args.max_min_qpe[1]) / (self.args.max_min_qpe[0] - self.args.max_min_qpe[1])

                L1_data = torch.nan_to_num(L1_data, nan=0.0).clamp(min=0.0)
                L1_data.sub_(self.args.max_min[1])
                L1_data.div_(self.args.max_min[0] - self.args.max_min[1])

                inputs = torch.cat([L1_data, qpe_data[:, :self.args.in_len_val,
                         ...]], dim=2)

                time_data = time_data.type(torch.float32).repeat(inputs.shape[0], 1, 1).to(self.device, non_blocking=True)

                geo_coords = torch.stack([self.geo_coords[:, lat_idx+self.args.border_tar[0]:lat_idx + self.coords_seg_size[0]-self.args.border_tar[0],
                                            lon_idx+self.args.border_tar[1]:lon_idx + self.coords_seg_size[1]-self.args.border_tar[1]] for (lat_idx, lon_idx) in self.test_indices]).to(non_blocking=True)
                dem_data = torch.stack([self.dem_data[:, int(
                    self.args.dem_ratio[0] * (lat_idx+self.args.border_tar[0])):int(
                    self.args.dem_ratio[0] * (lat_idx + self.coords_seg_size[0]-self.args.border_tar[0])),
                                        int(self.args.dem_ratio[1] * (lon_idx+self.args.border_tar[1])):int(
                                            self.args.dem_ratio[1] * (lon_idx + self.coords_seg_size[1]-self.args.border_tar[1]))] for (lat_idx, lon_idx) in self.test_indices]).to(
                    non_blocking=True)

                time0 = time.time()

                chunk_size = 1
                pred, pred_class = torch.empty(0, forcast_len, *qpe_data.shape[-3:]) \
                    , torch.empty(0, forcast_len, *qpe_data.shape[-3:])#.to(self.device)
                for i in range(int(inputs.shape[0] / chunk_size)):
                    # print(i)
                    pred_tmp, pred_class_tmp = self.model(inputs[i*chunk_size:(i+1)*chunk_size].clone(), era5_data[i*chunk_size:(i+1)*chunk_size].clone()
                                                          , self.const_data, time_data[i*chunk_size:(i+1)*chunk_size].clone(),
                                                          geo_coords=geo_coords[i*chunk_size:(i+1)*chunk_size].clone(),
                                                          dem_data=dem_data[
                                                                     i * chunk_size:(i + 1) * chunk_size].clone(),
                                                  aft_seq_length=forcast_len,
                                                  shrink=self.args.shrink, mode=mode,
                                                  device=self.device)
                    pred = torch.cat([pred, pred_tmp.cpu()])
                    pred_class = torch.cat([pred_class, pred_class_tmp.cpu()])

                mask = (pred_class > 0).type(torch.int)
                last_res = (pred * mask).to(self.device)
                last_res = self.merge_patches_efficient(
                    last_res.unflatten(0, [bs, bs_p]), [self.max_indices[0]+self.args.tar_size[0]-2*self.args.border_tar[0],
                                                        self.max_indices[1]+self.args.tar_size[1]-2*self.args.border_tar[1]]
                )

                time_list.append((time.time() - time0))
                print(last_res.shape)

                # 将 label 裁剪到与 last_res 相同的空间尺寸
                label_plot = labels[:, :, :,
                             self.args.border_tar[0]: last_res.shape[-2] + self.args.border_tar[0],
                             self.args.border_tar[1]: last_res.shape[-1] + self.args.border_tar[1]]

                save_dir = os.path.join(self.results_dir, 'qualitative_maps')
                print(torch.max(last_res))
                print(torch.max(label_plot))

                self.plot_qualitative_step(
                    label=label_plot,
                    pred=last_res,
                    step=step,
                    time0=self.time0[step],
                    save_dir=save_dir,
                    max_min_qpe=self.args.max_min_qpe
                )

    def plot_qualitative_step(self, label, pred, step, time0, save_dir,
                              max_min_qpe=None,
                              lat_range=(29.6, 44.4),
                              lon_range=(250.8, 285.2)):
        """
        绘制 4 行 × 2 列 定性对比图。
        布局：
            行0: 真值 +6h  |  真值 +12h
            行1: 预测 +6h  |  预测 +12h
            行2: 真值 +18h |  真值 +24h
            行3: 预测 +18h |  预测 +24h
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.gridspec import GridSpec
        from matplotlib.cm import ScalarMappable
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import torch
        import os

        # ---------- 数据预处理 ----------
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        label = np.squeeze(label)
        pred = np.squeeze(pred)

        selected_items = [
            (5, 6),  # +6h
            (11, 12),  # +12h
            (17, 18),  # +18h
            (23, 24),  # +24h
        ]
        T = label.shape[0]
        assert T > 23, f"需要至少 24 个时刻，当前只有 {T}"

        # ---------- MRMS QPE 颜色映射 ----------
        colors = [
            [1, 1, 1], [0, 0.6, 1], [0, 1, 0], [1, 1, 0],
            [1, 0.6, 0], [1, 0, 0], [0.8, 0.4, 0.8]
        ]
        boundaries = [0, 0.2, 1, 2, 4, 8, 20, 50]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        # new_h, new_w = label.shape[1:]
        # # 创建图形
        # fig_width = (new_w) / 100
        # fig_height = (new_h) / 100
        # ---------- 图形布局 ----------
        fig = plt.figure(figsize=(20, 22), dpi=150)
        gs = GridSpec(4, 2, figure=fig,
                      wspace=0.03, hspace=0.06,
                      left=0.10, right=0.94, top=0.93, bottom=0.08)  # 增大 bottom 预留 colorbar

        extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
        lat_ticks = [30, 35, 40]
        lon_ticks_w = [100, 90, 80]
        lon_ticks_raw = [360 - w for w in lon_ticks_w]

        # ---------- 遍历绘制 ----------
        for row_idx in range(4):
            group_idx = row_idx // 2
            method_idx = row_idx % 2
            method_name = 'Observation' if method_idx == 0 else 'Ours'

            for col_idx in range(2):
                item_idx = group_idx * 2 + col_idx
                data_idx, hour = selected_items[item_idx]

                if data_idx >= T:
                    continue

                ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
                ax.set_extent(extent, crs=ccrs.PlateCarree())

                # 强制子图充满整个 Axes，避免因地图比例被压缩
                ax.set_aspect('auto')

                # 黑色粗边框
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(3)

                # 地图底图
                ax.add_feature(cfeature.STATES.with_scale('110m'), linewidth=0.6,
                               edgecolor='#444444', alpha=0.9, facecolor='none')
                ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.6,
                               edgecolor='#444444', alpha=0.9)
                ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.5,
                               edgecolor='#444444', alpha=0.9)

                # ---- 获取二维降水数据 ----
                data = label[data_idx] if method_idx == 0 else pred[data_idx]
                H, W = data.shape

                # ---- 构造经纬度网格（假设网格纬度从北向南递减） ----
                # 根据 lat_range 的方向确定纬度数组
                lat_vals = np.linspace(lat_range[1], lat_range[0], H)  # 高纬度在上
                lon_vals = np.linspace(lon_range[0], lon_range[1], W)

                # ---- 使用 pcolormesh 绘制降水数据 ----
                ax.pcolormesh(lon_vals, lat_vals, data,
                              transform=ccrs.PlateCarree(),
                              cmap=cmap, norm=norm,
                              shading='auto', rasterized=True)

                # ---- 子图内左上角标注时刻 ----
                ax.text(0.03, 0.97, f'+{hour} h', transform=ax.transAxes,
                        fontsize=13, fontweight='bold', va='top', ha='left',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='black', linewidth=1.5, alpha=0.9))

                # ---- 左侧方法名（竖排）----
                if col_idx == 0:
                    ax.text(-0.12, 0.5, method_name, transform=ax.transAxes,
                            fontsize=15, fontweight='bold', va='center', ha='right',
                            rotation=90, rotation_mode='anchor')

                # ---- 外侧经纬度刻度 ----
                # if row_idx == 3:
                #     ax.set_xticks(lon_ticks_raw, crs=ccrs.PlateCarree())
                #     ax.set_xticklabels([f'{w}°W' for w in lon_ticks_w], fontsize=11)
                #     ax.xaxis.set_ticks_position('bottom')
                # else:
                #     ax.set_xticks([])

                # if col_idx == 0:
                #     ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                #     ax.set_yticklabels([f'{lat}°N' for lat in lat_ticks], fontsize=11)
                #     ax.yaxis.set_ticks_position('left')
                # else:
                #     ax.set_yticks([])

                # ---- 内侧经纬度白底标注 ----
                if col_idx == 0:
                    for lat in lat_ticks:
                        ratio = (lat_range[1] - lat) / (lat_range[1] - lat_range[0])
                        ax.text(0.04, 1.0 - ratio, f'{lat}°N',
                                transform=ax.transAxes, fontsize=10, color='black',
                                va='center', ha='left',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          edgecolor='none', alpha=0.85))

                if row_idx == 3:
                    for lon_raw, lon_w in zip(lon_ticks_raw, lon_ticks_w):
                        ratio = (lon_raw - lon_range[0]) / (lon_range[1] - lon_range[0])
                        ax.text(ratio, 0.04, f'{lon_w}°W',
                                transform=ax.transAxes, fontsize=10, color='black',
                                ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          edgecolor='none', alpha=0.85))

                # ---- 细网格线 ----
                ax.gridlines(draw_labels=False, xlocs=range(250, 290, 5),
                             ylocs=range(25, 50, 5), linewidth=0.3,
                             color='gray', alpha=0.4)

        # ---------- 总标题 ----------
        fig.suptitle(f'Time {time0}',
                     fontsize=18, fontweight='bold', y=0.97)

        # ---------- 水平 colorbar（底部，调整位置不与子图重叠）----------
        cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Precipitation Rate (mm / h)', fontsize=13)
        cbar.set_ticks([0, 0.2, 1, 2, 4, 8, 20, 50])
        cbar.ax.tick_params(labelsize=10)

        # ---------- 保存 ----------
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'time{time0}_step{step:04d}_quali.png')
        plt.savefig(save_path, dpi=150, edgecolor='none')
        plt.close()
        print(f"[Qualitative] Saved: {save_path}")


