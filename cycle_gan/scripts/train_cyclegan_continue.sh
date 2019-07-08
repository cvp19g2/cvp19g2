set -ex
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot /cvhci/data/praktikumSS2019/gruppe2/CelebA/classified_aligned/train_white_black --name new_images_aligned_30 --model cycle_gan --pool_size 50 --no_dropout --continue_train
