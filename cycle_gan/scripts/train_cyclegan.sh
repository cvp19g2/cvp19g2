set -ex
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot /home/p19g2/data_race_add/train --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
