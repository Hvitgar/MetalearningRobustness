python3 ~/Metalearning-Robustness/code/train.py -a resnet18 -j 16 --gpu 0 --effective-bs 128 --evaluate corrupted --augment-train-data --increasing-alpha 50 --augmentations Identity StyleTransfer --augmentation-mean 0 --min-magnitude 0 --max-magnitude 1 --resume ../../baseline_cont/model_best.pth.tar --epochs 850 --seed 131213 --style-subset 128 --train /gpfs01/bethge/data/sixteen-class-imagenet-balanced-trainval
