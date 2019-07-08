set -ex
python test.py --dataroot ./data/test/ --name maps_cyclegan --model cycle_gan --phase test --no_dropout
