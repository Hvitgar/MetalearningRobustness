python3 ~/Metalearning-Robustness/code/train.py -a resnet18 -j 16 --gpu 0 --effective-bs 128 --evaluate corrupted --augment-train-data --increasing-alpha --augmentations Identity StyleTransfer --augmentation-mean 0.9 --min-magnitude 0 --max-magnitude 1 --resume checkpoint_alpha_0.900.pth.tar --epochs 650 --seed 131213 --train /gpfs01/bethge/data/sixteen-class-imagenet-balanced-trainval
