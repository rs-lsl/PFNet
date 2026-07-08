# -*- coding: utf-8 -*-
import random
from functools import partial
from itertools import repeat
from typing import Callable
import xarray as xr
import matplotlib.pyplot as plt
# from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
# import cf
# import cfdm
import torch.utils.data
import numpy as np
from osgeo import gdal
import numpy as np
import torch
import torch.nn.functional as F
# from scipy.interpolate import griddata
# import esmpy
#  python weatherbench2_main/model2023/datasets/utils_dataset.py
import torch
import torch.nn.functional as F

"""
PyTorch 版增量 PCA（替代 sklearn IncrementalPCA）
解决 LAPACK 整数溢出问题，支持 GPU/CPU
"""

import torch
import numpy as np
import joblib

def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件

    # im_width =   # 栅格矩阵的列数
    # im_height =   # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    return dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    # return im_data

# 读图像文件
def read_img_gdal(filename):
    dataset = gdal.Open(filename)  # 打开文件

    # im_width =   # 栅格矩阵的列数
    # im_height =   # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    return dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    # return im_data


# 写文件，以写成tif为例
def write_img_gdal(filename, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影
    # print(dataset)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


# 读图像文件
def read_img_gdal(filename):
    dataset = gdal.Open(filename)  # 打开文件

    # im_width =   # 栅格矩阵的列数
    # im_height =   # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    return dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    # return im_data


# 写文件，以写成tif为例
def write_img_gdal(filename, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影
    # print(dataset)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def read_img(filename):
    # 读取tif影像
    dataset = gdal.Open(filename)  # "H:/worldview2/result/origin_pan.tif"
    # print(dataset)
    im_cols = dataset.RasterXSize  # 栅格矩阵的列数
    im_rows = dataset.RasterYSize  # 栅格矩阵的行数
    # im_bands = dataset.RasterCount

    # band = dataset.GetRasterBand(1)
    im_data = np.float32(dataset.ReadAsArray(0, 0, im_cols, im_rows))  # 获取数据

    return im_data

def read_img_gdal(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    return im_data


# 写文件，以写成tif为例
def write_img_gdal(filename, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def fast_collate_for_prediction(batch):
    """ A fast collation function optimized for float32 images (np array or torch)
        and float32 targets (video prediction labels) in video prediction tasks"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into
        # one tensor ordered by position such that all tuple of position n will end up
        # in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.float32)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            # all input tensor tuples must be same length
            assert len(batch[i][0]) == inner_tuple_size
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.zeros((batch_size, *batch[1][0].shape), dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x

# from prefetch_generator import BackgroundGenerator
# class DataLoaderX(torch.utils.data.DataLoader):
#
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

class PrefetchLoader3:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_input2, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input2 = next_input2.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_input2 = next_input2.float()
                        next_target = next_target.float()

            if not first:
                yield input, input2, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            input2 = next_input2
            target = next_target

        yield input, input2, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class PrefetchLoader4:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_input2, next_input3, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input2 = next_input2.cuda(non_blocking=True)
                next_input3 = next_input3.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_input2 = next_input2.half()
                        next_input3 = next_input3.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_input2 = next_input2.float()
                        next_input3 = next_input3.float()
                        next_target = next_target.float()

            if not first:
                yield input, input2, input3, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            input2 = next_input2
            input3 = next_input3
            target = next_target

        yield input, input2, input3, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class PrefetchLoader5:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_input2, next_input3, next_input4, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input2 = next_input2.cuda(non_blocking=True)
                next_input3 = next_input3.cuda(non_blocking=True)
                next_input4 = next_input4.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_input2 = next_input2.half()
                        next_input3 = next_input3.half()
                        next_input4 = next_input4.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_input2 = next_input2.float()
                        next_input3 = next_input3.float()
                        next_input4 = next_input4.float()
                        next_target = next_target.float()

            if not first:
                yield input, input2, input3, input4, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            input2 = next_input2
            input3 = next_input3
            input4 = next_input4
            target = next_target

        yield input, input2, input3, input4, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class PrefetchLoader6:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_input2, next_input3, next_input4, next_input5, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input2 = next_input2.cuda(non_blocking=True)
                next_input3 = next_input3.cuda(non_blocking=True)
                next_input4 = next_input4.cuda(non_blocking=True)
                next_input5 = next_input5.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_input2 = next_input2.half()
                        next_input3 = next_input3.half()
                        next_input4 = next_input4.half()
                        next_input5 = next_input5.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_input2 = next_input2.float()
                        next_input3 = next_input3.float()
                        next_input4 = next_input4.float()
                        next_input5 = next_input5.float()
                        next_target = next_target.float()

            if not first:
                yield input, input2, input3, input4, input5, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            input2 = next_input2
            input3 = next_input3
            input4 = next_input4
            input5 = next_input5
            target = next_target

        yield input, input2, input3, input4, input5, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class PrefetchLoader7:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_input2, next_input3, next_input4, next_input5, next_input6, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input2 = next_input2.cuda(non_blocking=True)
                next_input3 = next_input3.cuda(non_blocking=True)
                next_input4 = next_input4.cuda(non_blocking=True)
                next_input5 = next_input5.cuda(non_blocking=True)
                next_input6 = next_input6.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_input2 = next_input2.half()
                        next_input3 = next_input3.half()
                        next_input4 = next_input4.half()
                        next_input5 = next_input5.half()
                        next_input6 = next_input6.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_input2 = next_input2.float()
                        next_input3 = next_input3.float()
                        next_input4 = next_input4.float()
                        next_input5 = next_input5.float()
                        next_input6 = next_input6.float()
                        next_target = next_target.float()

            if not first:
                yield input, input2, input3, input4, input5, input6, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            input2 = next_input2
            input3 = next_input3
            input4 = next_input4
            input5 = next_input5
            input6 = next_input6
            target = next_target

        yield input, input2, input3, input4, input5, input6, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input in self.loader:
            with torch.cuda.stream(stream):
                for i in range(len(next_input)):
                    next_input[i] = next_input[i].cuda(non_blocking=True)
                # next_input = next_input.cuda(non_blocking=True)
                # next_input2 = next_input2.cuda(non_blocking=True)
                # next_input3 = next_input3.cuda(non_blocking=True)
                # next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_input2 = next_input2.half()
                        next_input3 = next_input3.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        for i in range(len(next_input)):
                            next_input[i] = next_input[i].float()
                        # next_input = next_input.float()
                        # next_input2 = next_input2.float()
                        # next_input3 = next_input3.float()
                        # next_target = next_target.float()

            if not first:
                yield next_input
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            # input2 = next_input2
            # input3 = next_input3
            # target = next_target

        yield input#, input2, input3, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(dataset,
                  batch_size,
                  shuffle=True,
                  is_training=False,
                  mean=None,
                  std=None,
                  num_workers=1,
                  num_aug_repeats=0,
                  input_channels=1,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=True,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=False,
                  worker_seeding='all',
                  return_num=4):
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                # sampler = torch.utils.data.BatchSampler(
                #     sampler, batch_size, drop_last=True
                # )
                print('setting the dis sampler')
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats==0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if use_prefetcher:
        # loader = PrefetchLoader(
        #     loader,
        #     mean=mean,
        #     std=std,
        #     channels=input_channels,
        #     fp16=fp16,
        # )
        if return_num==3:
            loader = PrefetchLoader3(
                loader,
                mean=mean,
                std=std,
                channels=input_channels,
                fp16=fp16,
            )
        elif return_num==4:
            loader = PrefetchLoader4(
                loader,
                mean=mean,
                std=std,
                channels=input_channels,
                fp16=fp16,
            )
        elif return_num==5:
            loader = PrefetchLoader5(
                loader,
                mean=mean,
                std=std,
                channels=input_channels,
                fp16=fp16,
            )
        elif return_num==6:
            loader = PrefetchLoader6(
                loader,
                mean=mean,
                std=std,
                channels=input_channels,
                fp16=fp16,
            )
        elif return_num==7:
            loader = PrefetchLoader7(
                loader,
                mean=mean,
                std=std,
                channels=input_channels,
                fp16=fp16,
            )

    # for images, labels in loader:
    #     print(images.shape)
    # print(loader)
    return loader, sampler

if __name__ == '__main__':
    pass