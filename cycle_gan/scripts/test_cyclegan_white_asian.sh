set -ex
CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /cvhci/data/praktikumSS2019/gruppe2/CelebA/classified_aligned/train_white_asian --name asian_test --model cycle_gan --phase test --no_dropout --num_test 2000
