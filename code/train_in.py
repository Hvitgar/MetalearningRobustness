import argparse
import os
import random
import shutil
import time
import warnings
import pickle
import imagecorruptions as ic
import numpy as np
from PIL import Image
import glob
import kornia

import augmentations
from augmentations import AdaptiveStyleTransfer

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from models.AugModel import AugModel
from datasets import InfiniteDataLoader
import sys
sys.path.append('/gpfs01/bethge/home/bmitzkus/src/hypergradient')
from roland import calculate_grad_g

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
augmentation_names = sorted(name for name in augmentations.__dict__ if not name.startswith("__") and callable(augmentations.__dict__[name]))

# argument parser
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--effective-bs', default=256, type=int, help='number of images that fit in GPU memory (must be a divisor of batch size)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='clean',
                    choices=['clean', 'corrupted'], help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=None, type=int, nargs='+',
                    help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='Set seed for deterministic data splitting.')
parser.add_argument('--train', action='store_true', help='set true to train the model')
parser.add_argument('--increasing-alpha', type=int, default=None, help='if set, increase alpha by 0.1 every n epochs')
parser.add_argument('--style-subset', default=1024, type=int, help='set this parameter to only use a random subset of styles')

def main():
    best_acc1 = 0
    args = parser.parse_args()
    assert args.batch_size % args.effective_bs == 0, "Effective batch size must be a divisor of batch_size"
    
    if len(args.gpu) < 2:
        print("Two GPU's are needed for training. Exiting...")
        exit()
    else:
        print("Use GPU: {} for training".format(args.gpu[:2]))
        gpu0 = args.gpu[0]
        gpu1 = args.gpu[1]
        
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        
    model = model.cuda(gpu0)
        
    if args.train:
        writer = SummaryWriter()
        optimizer = torch.optim.SGD(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss().cuda(gpu0)
        
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu0)
            checkpoint = torch.load(args.resume, map_location=loc)
            
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.to(gpu0)            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            if args.train:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded optimizer state from checkpoint")
                except:
                    print("=> optimizer state not found in checkpoint")
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    cudnn.benchmark = True
    
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    norm_params = {'mean':[0.485, 0.456, 0.406],
                  'std':[0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(mean=norm_params['mean'],
                                     std=norm_params['std'])
    NORM = kornia.color.Normalize(mean=torch.tensor(norm_params['mean'], dtype=torch.float32), std=torch.tensor(norm_params['std'], dtype=torch.float32))
    test_loader = get_test_loader(args, normalize)
    if args.evaluate == 'corrupted':
        corrupted_test_loader = get_test_loader(args, normalize, lambda img: apply_random_corruption(img, test=True))
            
    
    if args.train:
        # as augmodel will be applied before normalization,
        AST = AdaptiveStyleTransfer(temperature=torch.tensor(0.1), mean=0., logits=torch.zeros(args.style_subset, dtype=torch.float32, requires_grad=True))
        AST.cuda(gpu1)
        AST.initStyles(args.style_subset, seed=args.seed)
        
        def styletransfer(img):
            if np.random.uniform() < 0.5:
                img = AST(img)
            return NORM(img)
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
        ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)
        
        # training
        for epoch in range(args.start_epoch, args.epochs):
            if args.increasing_alpha is not None and (epoch - args.start_epoch) % args.increasing_alpha == 0:
                current_alpha = AST.mu_mag
                
                ckpt = {
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }
                save_checkpoint(ckpt, is_best=False, filename='checkpoint_alpha_%1.3f.pth.tar'%(current_alpha.item()))
                
                updated_alpha = current_alpha + 0.1
                AST.mu_mag = updated_alpha
                print('=> Alpha={}'.format(AST.mu_mag.item()))
            train(train_loader, model, styletransfer, criterion, optimizer, epoch, args, writer)
            is_best = False
            # evaluate on validation set
            if epoch % args.print_freq == 0:
                acc1 = validate(test_loader, model, criterion, args)
                writer.add_scalar('Metrics/test_acc', acc1, epoch)
                if args.evaluate == 'corrupted':
                    mpc = validate(corrupted_test_loader, model, criterion, args)
                    writer.add_scalar('Metrics/test_mpc', mpc, epoch)
                
                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
            
            ckpt = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }
            

            save_checkpoint(ckpt, is_best)
            writer.add_scalar('AST/alpha', AST.mu_mag.cpu().detach().numpy(), epoch)

            
    if args.evaluate == 'clean':
        validate(test_loader, model, criterion, args)
    elif args.evaluate == 'corrupted':
        corruptions = ic.get_corruption_names('all')
        severities = [0,1,2,3,4,5]
        accuracies = {}
        for corruption in corruptions:
            accuracies[corruption] = {}
            for severity in severities:
                if severity == 0:
                    print('Testing clean')
                    acc = validate(test_loader, model, criterion, args)
                    accuracies[corruption][severity] = torch.squeeze(acc.cpu()).item()
                else:
                    print('Testing %s:%d'%(corruption, severity))
                    corrupted_loader = get_test_loader(args, normalize, lambda x: Image.fromarray(ic.corrupt(np.array(x, dtype=np.uint8), corruption_name=corruption, severity=severity)))
                    acc = validate(corrupted_loader, model, criterion, args)
                    accuracies[corruption][severity] = torch.squeeze(acc.cpu()).item()
        if args.train:
            e = args.epochs
        elif args.resume:
            e = args.start_epoch
        pickle.dump(accuracies, open("robustness_epoch_{}.pkl".format(e), "wb"))
    
def train(train_loader, model, styletransfer, criterion, optimizer, epoch, args, writer):
    losses = AverageMeter('Loss', ':.4e')
    val_losses = AverageMeter('ValLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    val_top1 = AverageMeter('ValAcc@1', ':6.2f')
    val_top5 = AverageMeter('ValAcc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5, val_losses, val_top1, val_top5],
        prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        splits = list(zip(*[torch.split(d, args.effective_bs) for d in batch])) # split batch, use gradient accumulation
        for images, target in splits:
            images = styletransfer(images)
            images = images.cuda(args.gpu[0], non_blocking=True)
            target = target.cuda(args.gpu[0], non_blocking=True)

            output = model(images)
            train_loss = criterion(output, target)
            ratio = len(images) / len(batch[0])
            weighted_train_loss = train_loss * ratio # weighting of the loss for gradient accumulation

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(train_loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            weighted_train_loss.backward()
        optimizer.step()
                
        if i % args.print_freq == 0:
            progress.display(i)
            
    # write to tensorboard
    writer.add_scalar('Losses/train', losses.avg, epoch)
    writer.add_scalar('Metrics/train_acc1', top1.avg, epoch)
    writer.add_scalar('Metrics/train_acc5', top5.avg, epoch)
        
    
def get_test_loader(args, normalize, pre_normalize_fn=None):
    testdir = os.path.join(args.data, 'val')
    if pre_normalize_fn is None:
        pre_normalize_fn = lambda x: x
        
    loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Lambda(pre_normalize_fn),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    return loader
        

def apply_random_corruption(img, test=False):
    severity = np.random.choice(6)
    if severity == 0:
        return img
    else:
        img = np.array(img).astype(np.uint8)
        if test:
            corruption = np.random.choice(ic.get_corruption_names())
        else:
            corruption = np.random.choice(ic.get_corruption_names('validation'))
        corrupted = ic.corrupt(img, severity=severity, corruption_name=corruption)
        corrupted = Image.fromarray(corrupted)
        return corrupted
    
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu[0], non_blocking=True)
            target = target.cuda(args.gpu[0], non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def stylized_loader(path):
    directory, basename = os.path.split(path)
    fname, ext = os.path.splitext(basename)
    trainval, classname = os.path.split(directory)
    classid = fname.split('_')[0]
    orig_root, split = os.path.split(trainval)
    stylized_root = '/gpfs01/bethge/data/imagenet-styletransfer/'
    stylized_path = os.path.join(stylized_root, split, classid, fname+'.png')
    return datasets.folder.default_loader(stylized_path)
    # code below works for train+val split, but is slow due to glob.glob
    # directory, basename = os.path.split(path)
    # fname, ext = os.path.splitext(basename)
    # stylized_root = '/gpfs01/bethge/data/imagenet-styletransfer/'
    # print(stylized_root + '**/' + fname + '.png')
    # candidates = glob.glob(stylized_root + '**/' + fname + '.png', recursive=True)
    # assert len(candidates) == 1, 'Found %d matches for file %s' % (len(candidates), path)
    # return datasets.folder.default_loader(candidates[0])


if __name__ == '__main__':
    main()
