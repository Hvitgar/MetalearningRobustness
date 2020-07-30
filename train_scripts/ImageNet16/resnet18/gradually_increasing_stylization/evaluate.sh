python3 ~/Metalearning-Robustness/code/train.py -a resnet18 -j 16 --gpu 0 --effective-bs 128 --evaluate corrupted --resume $1 /gpfs01/bethge/data/sixteen-class-imagenet-balanced-trainval
