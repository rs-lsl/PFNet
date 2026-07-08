# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
# from torchvision.datasets import mnist
from torch.autograd import Variable
from tqdm import tqdm
import sys
import tempfile
import time

# import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
# from model2023.metrics import metric
from datetime import timedelta


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # print("NOT using distributed mode")
        raise EnvironmentError("NOT using distributed mode")
        # return
    # print(args)
    #
    args.distributed = True

    # 这里需要设定使用的GPU
    torch.cuda.set_device(args.gpu)
    # 这里是GPU之间的通信方式，有好几种的，nccl比较快也比较推荐使用。
    args.dis_backend = 'nccl'
    # 启动多GPU
    dist.init_process_group(
        backend=args.dis_backend,
        init_method=args.dis_url,
        world_size=args.world_size,
        # timeout=timedelta(seconds=7200000),
        rank=args.rank,
        device_id=torch.device(f"cuda:{args.gpu}")
    )
    # 这个是：多GPU之间进行同步，也就是有的GPU跑的快，有的跑的慢（比如当你判断if RANK == 0: do something， 那么进程0就会多执行代码速度慢）
    # 所以这个代码就是等待所有进程运行到此处。
    dist.barrier()


def init_distributed_mode_old(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # print("NOT using distributed mode")
        raise EnvironmentError("NOT using distributed mode")
        # return
    # print(args)
    #
    args.distributed = True

    # 这里需要设定使用的GPU
    torch.cuda.set_device(args.gpu)
    # 这里是GPU之间的通信方式，有好几种的，nccl比较快也比较推荐使用。
    args.dis_backend = 'nccl'
    # 启动多GPU
    dist.init_process_group(
        backend=args.dis_backend,
        init_method=args.dis_url,
        world_size=args.world_size,
        # timeout=timedelta(seconds=7200000),
        rank=args.rank
    )
    # 这个是：多GPU之间进行同步，也就是有的GPU跑的快，有的跑的慢（比如当你判断if RANK == 0: do something， 那么进程0就会多执行代码速度慢）
    # 所以这个代码就是等待所有进程运行到此处。
    dist.barrier()


def cleanup():
    # 这里不同我多说，看名字就知道啥意思
    dist.destroy_process_group()

# 判断多GPU是否启动
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
# 拿到你有几个GPU，数量。主要是用来all_reduce计算的。
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# 拿到进程的rank
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# 这个主要就是和all_reduce差不多，每一个进程都会算到一个值，把不同进程的值回合起来。
# 比如对于loss而言，进程0是1、2样本得到的loss是0.1，进程1是3、4样本得到的loss是0.2，那么
# 该批次的all_loss就是0.1+0.2=0.3
# 同样 你在计算分类正确时候 比如batch是100，进程0得到的正确个数是50 进程1得到的正确率60，那么整体的准确率就是110/200
def reduce_value(value, average=True):
    # 拿到GPU个数，主要是判断我们有几个进程
    world_size = get_world_size()
    # 如果单进程就返回
    if world_size < 2:
        return value

    with torch.no_grad():
        # 这个就是all_reduce把不同进程的值都汇总返回。
        dist.all_reduce(value)
        if average:
            # 是否取均值
            value /= world_size
        return value

# 判断是否是主进程，主进程的意思就是rank=0，
# 严格意义上来说没有主进程之分，你想进程1是主进程，那么你就 get_rank() == 1就行。
def is_main_process():
    return get_rank() == 0

def clip_grads(params, args, norm_type: float = 2.0):
    """ Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    args.clip_mode = args.clip_mode if args.clip_grad is not None else None
    if args.clip_mode is None:
        return
    if args.clip_mode == 'norm':
        torch.nn.utils.clip_grad_norm_(params, args.clip_grad, norm_type=norm_type)
    elif args.clip_mode == 'value':
        torch.nn.utils.clip_grad_value_(params, args.clip_grad)
    else:
        assert False, f"Unknown clip mode ({args.clip_mode})."


if __name__ == '__main__':
    pass