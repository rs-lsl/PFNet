# -*- coding: utf-8 -*-
import os
import os.path as osp
import numpy as np
# import xarray as xr
import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings('ignore')

import torch.distributed as dist
import time
from parser import create_parser

from DDP import init_distributed_mode, cleanup, reduce_value, clip_grads, init_distributed_mode_old

from utils0 import create_folder_if_not_exists, copy_all_files, save_command

if __name__ == '__main__':
    time0 = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    import torch
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.backends.cudnn.benchmark = True

    args = create_parser().parse_args()
    config = args.__dict__
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")

    init_distributed_mode_old(args)

    rank = args.rank
    batch_size = args.batch_size

    # 获得gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    global device
    device = torch.device("cuda", local_rank)

    if rank == 0:
        save_command()
        print('args.world_size', args.world_size)
        print('rank', rank)
        print('device', device)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    args.dataname = 'ERA5_Nexrad_mrms_qpe_2001_2017_usa'
    if args.dataname == 'ERA5_Nexrad_mrms_qpe_2001_2017_usa':
        from dataloader import load_ERA5_nexrad_mrms_qpe_2001_2017_usa_2km  # , write_zarr, load_ERA5_dataset_zarr
        from DDP_era5_nexrad_mrms_usa_2km import Pred_model#, Pred_model_auto
        from model_era5_nexrad_qpe_usa_2km import SimVP_Model_x  # , Discriminator, Encoder, Decoder

        args.minlat = None
        args.pre_seq_length = 1  # per output of the model
        args.aft_seq_length_val = 24  # total time length to be predicted
        args.aft_seq_length_test = 24  # total time length to be predicted, set to 60 for insert to fuxi
        args.input_time_length = 2  # input time length of the model,
        args.in_len_val = 5
        args.shrink = 1 if args.input_time_length > args.pre_seq_length else 0
        args.time_emb_num = 42  # mounth/day/hour/minute

        args.resume_epoch = None   #  None
        args.compute_mean_std = False

        args.lat_dismiss = [0, 0]
        args.lon_dismiss = [0, 0]
        args.eval_idx = 0
        args.SNR_scale = 0.01
        args.warmup_epoch = 20
        args.product_name = ['Z_H', 'AzShr', 'Div', 'K_DP', 'SW', 'Z_DR', 'r_HV']
        args.threholds = np.array([0.2, 1, 2, 4, 8, 20])  # ****************************  20,30,40
        args.BALANCING_WEIGHTS = (1, 1, 1, 1, 1, 1, 1)  # 1 1 4 8
        args.sample_inte = 1
        args.sample_inte_val = 6
        args.sample_inte_test = 6
        args.ori_thre = np.min(args.threholds)  # to classify if the pixel is with raining in the original data
        print('args.ori_thre', args.ori_thre)
        args.border_tar = [256, 256]
        args.tar_size = [512+2*args.border_tar[0], 512+2*args.border_tar[1]]
        args.crop_stride_test = [1200-512-256-256-1, 250]
        args.H_d, args.W_d = 96+12, 96+12  # hidden state size ****************  96,96
        args.trainset_ratio = 0.8
        args.valset_ratio = 0.1
        args.weight_decay = 1e-1
        args.drop = 0.05

        args.time_inte = [1, 2, 4]
        args.iter_len_epoch = [0, 300, 500]  # ************
        args.pred_len = list(range(2, 6, 2))  # ************
        args.batch_size_list = [7, 6]
        assert args.iter_len_epoch[-1] <= args.epoch
        assert len(args.pred_len) == len(args.iter_len_epoch) - 1
        assert args.aft_seq_length_test >= max(args.pred_len) * max(args.time_inte)

        hid_S, hid_T, N_S, N_T = 24, 256, 3, 3  # hid_S and N_S**************
        args.hid_S = hid_S
        args.hid_const = 4
        loss_type_adv = 'binary_cross_entropy'

        # please modify these two pathes
        args.little_file_path = '/home/lisl/weatherbench2_main/model_prob/PFNet/files/'
        data_root_dir = '/data02/lisl/'
        results_dir = os.path.join(data_root_dir, 'results', 'results_' + args.dataname)
        num_workers = 4
        L1_data_path = os.path.join(data_root_dir, 'radar/3D_NEXRAD/2016_2017_1hour/nexrad_2001-2017_single_chunk.zarr')
        era5_data_path = os.path.join(data_root_dir, 'era5_post/era5_usa_2001_2017_with_new_vars.zarr')  #  _with_new_vars
        mrms_qpe_path = os.path.join(data_root_dir, 'MRMS/qpe_2001-2017.zarr')  #
        dem_path = os.path.join(args.little_file_path, 'American_clip_region_2km.tif')
        cp_dir = os.path.join(results_dir, 'checkpoints/', args.ex_name)
        args.save_dir = os.path.join(results_dir, 'logs', args.ex_name)
        if rank == 0:
            create_folder_if_not_exists(cp_dir)
            create_folder_if_not_exists(os.path.join(results_dir, 'logs', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'quanti_figures', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'csv_results', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'quali_figures', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'results_show', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'results_hist', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'latent_3d_map', args.ex_name))
        if rank == 0 and 0:
            # write_zarr(save_dir)
            # write_ERA5_dataset(dataset_dir=dataset_dir, base_dir=save_dir, args=args)
            write_ERA5_dataset_per(dataset_dir=dataset_dir, save_dir=save_dir, args=args)
            exit()

        const_data = None  # np.load(osp.join(save_dir, 'var_const_data.npy'))
        args.max_min = [94.4649, 0]
        args.max_min_qpe = [10, 0]
        args.min_max_ori_thre = (args.ori_thre - args.max_min_qpe[1]) / (args.max_min_qpe[0] - args.max_min_qpe[1])
        args.out_ch = 1
        args.L1_shape = [13, 1200, 2300]  # 24+1
        args.era5_shape = [243+37*4, 97, 185]  #243+37*4
        tmp_scale_i = args.L1_shape[1] / args.era5_shape[1]  # 1200 / 97 ≈ 12.371
        tmp_scale_j = args.L1_shape[2] / args.era5_shape[2]  # 2300 / 185 ≈ 12.432
        args.tar_era5_shape = [args.era5_shape[0], int(args.tar_size[0] / tmp_scale_i) + 1,
                               int(args.tar_size[1] / tmp_scale_j) + 1]
        args.tar_L1_shape = [args.L1_shape[0], args.tar_size[0] - 2 * args.border_tar[0],
                             args.tar_size[1] - 2 * args.border_tar[1]]
        in_shape = [args.batch_size, args.input_time_length, *args.L1_shape]
        args.dem_shape = (1200, 2300)
        args.dem_ratio = (int(args.dem_shape[0] / args.L1_shape[-2]), int(args.dem_shape[1] / args.L1_shape[-1]))
        print('args.dem_ratio', args.dem_ratio)
        # args.dem_border = (int(args.border_tar[0]*args.dem_shape[0] / args.L1_shape[-2]), int(args.border_tar[1]*args.dem_shape[1] / args.L1_shape[-1]))

        per_read = True
        zarr_read = False
        dataloader_train, sampler_train, dataloader_val, dataloader_test = \
            load_ERA5_nexrad_mrms_qpe_2001_2017_usa_2km(batch_size=args.batch_size,
                                                    val_batch_size=args.val_batch_size,
                                                    test_batch_size=args.test_batch_size, lon_len=256, lat_len=256,
                                                    L1_data_path=L1_data_path, era5_data_path=era5_data_path,
                                                    mrms_qpe_path=mrms_qpe_path, file_path_dem=dem_path,
                                                    num_workers=num_workers, distributed=True,
                                                    use_prefetcher=True, test=args.test, data_root_dir=data_root_dir,
                                                    args=args)


    model = SimVP_Model_x(in_shape, out_ch=1, hid_S=hid_S, hid_T=hid_T, N_S=N_S, drop=args.drop,
                          spatio_kernel_enc=3,
                          spatio_kernel_dec=3, time_emb_num=args.time_emb_num, args=args, device=device).to(device)


    find_unused_parameters = True
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                           output_device=local_rank,
                                                           find_unused_parameters=find_unused_parameters)  # device[args.device]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)   # 把这一步放进pred_model中

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)

    # To replace the above, do the following:

    # from muon import MuonWithAuxAdam
    # # 使用 model.module 来访问原始模型
    # hidden_weights = [p for p in model.module.hid.parameters() if p.ndim >= 2]
    # hidden_gains_biases = [p for p in model.module.hid.parameters() if p.ndim < 2]
    # nonhidden_params = [*model.module.enc.parameters(), *model.module.dec.parameters(), *model.module.decoder_recon.parameters(),
    #                     *model.module.time_embedding.parameters()]
    # param_groups = [
    #     dict(params=hidden_weights, use_muon=True,
    #          lr=args.lr, weight_decay=args.weight_decay),
    #     dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
    #          lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay),
    # ]
    # optimizer = MuonWithAuxAdam(param_groups)

    if args.dataname == 'ERA5_Nexrad_mrms_qpe_2001_2017_usa':
        pred_model = Pred_model(model, optimizer, dataloader_train, sampler_train, dataloader_val, dataloader_test,
                                const_data, dem_path,
                                in_shape=[args.batch_size, args.input_time_length,
                                          *args.L1_shape], hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
                                time_emb_num=args.time_emb_num, results_dir=results_dir, device=device, rank=rank,
                                local_rank=local_rank, loss_type=loss_type_adv, cp_dir=cp_dir, args=args)
    else:
        pred_model = Pred_model(model, optimizer, dataloader_train, sampler_train, dataloader_val, dataloader_test,
                                const_data, #dem_path,
                                in_shape=[args.batch_size, args.input_time_length,
                                          *args.L1_shape], hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
                                time_emb_num=args.time_emb_num, results_dir=results_dir, device=device, rank=rank,
                                local_rank=local_rank, loss_type=loss_type_adv, cp_dir=cp_dir, args=args)

    pred_res = pred_model.test(mode='test')  # , test_epoch=test_epoch
