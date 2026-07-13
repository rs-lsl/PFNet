# PFNet
The code of "Accurate Precipitation Forecasting by Efficiently Learning from Massive Atmospheric Variables and Scarce Precipitation Samples".

To get the predicted results:

(1) Create a new environment by:

conda create -n py39 gdal python=3.9

pip install torch==2.0.1

pip install tqdm pandas matplotlib opencv-python lpips scikit-image numba zarr xarray

(2) Download the pretrained weight and datasets from: [https://www.kaggle.com/datasets/shuangliangli123/precip_era5_nexrad_mrms](https://www.kaggle.com/datasets/shuangliangli123/precip-era5-nexrad-mrms)

(3) Modify the paths of the dataset and each necessary file.

(4) Run the script: torchrun --nproc_per_node=1 --master_port=55568 weatherbench2_main/model_prob/PFNet/main_latent.py --epoch 500 --ex_name '260201_usa-precip-baseline' --batch_size 7 --val_batch_size 1 --test 1 --drop 0.0 --clip_grad 0 --eval_iter 1000 --save_iter 10 --test_epoch 100

Then the predicted results initialized from 2017-03-29 06:00:00 and 2017-10-29 12:00:00 will be generated in the qualitative_maps directory.

Should you have any questions, please feel free to contact me at whu_lsl@whu.edu.cn.

