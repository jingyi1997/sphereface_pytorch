from __future__ import print_function
import sys
sys.path.insert(0, '/mnt/lustre/xujingyi/.local/lib/python3.7/site-packages')
import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import distributed
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import time
import argparse
import numpy as np

from memcached_dataset import McDataset
from apex.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from utils import *
import net_sphere_2


parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--base_lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='')
parser.add_argument('--ckpt', default='experiment/softmax', type=str, help='')
parser.add_argument('--data_dir', default='/mnt/lustre/xujingyi/sphereface/preprocess/result/CASIA-WebFace-112X96', type=str)
parser.add_argument('--data_list', default='/mnt/lustre/xujingyi/sphereface/preprocess/result/webface.txt', type=str)
parser.add_argument('--loss_type', default='softmax', type=str)
parser.add_argument('--reg_weight', default=0.006, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--save_freq', default=1000, type=str)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--resume', default=None, type=int)
parser.add_argument('--lr_steps', default=[10, 15, 18], type=int)
parser.add_argument('--lr_mults', default=0.1, type=float)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()



def main():

    global  rank, world_size


    rank, world_size = dist_init("1234")

    if rank == 0:
      if not os.path.exists('{}/checkpoints'.format(args.ckpt)):
         os.makedirs('{}/checkpoints'.format(args.ckpt))
      if not os.path.exists('{}/logs'.format(args.ckpt)):
         os.makedirs('{}/logs'.format(args.ckpt))



    if rank == 0:
       logger = create_logger('global_logger', '{}/logs/{}.txt'.format(args.ckpt,time.time()))
       logger.info('{}'.format(args))
    else:
       logger = None
    # create model
    print("=> creating model '{}'".format(args.net))
    net = getattr(net_sphere_2,args.net)(head=args.loss_type)
    net = net.cuda()
    
    #net = nn.parallel.distributed.DistributedDataParallel(net)
    net = DDP(net)
    
    # build dataset
    data_dir = args.data_dir
    data_list = args.data_list
    
    train_dataset = McDataset(data_dir, data_list, transforms.Compose([
            transforms.RandomHorizontalFlip()]))

    train_sampler = distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    # build optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
 
    if args.loss_type == 'a-softmax':
        criterion = net_sphere_2.AngleLoss()
    if args.loss_type == 'softmax':
        criterion = torch.nn.CrossEntropyLoss()
    if args.loss_type == 'regular':
        criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    # optionally resume from a pretrained model
    if args.resume:
        model_name = 'epoch_'+str(args.resume)+'_ckpt.pth.tar'
        model_path = os.path.join(args.ckpt, 'checkpoints', model_name)
        checkpoint = torch.load(model_path)
        start_epoch = int(checkpoint['epoch'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    ## start training 
    net.train()
    prefetcher = DataPrefetcher(train_loader)
    freq = args.print_freq
    end = time.time()
    
    for epoch in range(start_epoch,args.epochs):
      train_sampler.set_epoch(epoch)
      train(net, epoch, train_loader, args, criterion, optimizer)
      if rank == 0:
             save_checkpoint({
               'epoch': epoch+1,
               'state_dict': net.state_dict(),
               'optimizer': optimizer.state_dict()
               },args,'epoch_'+str(epoch+1)+'_ckpt.pth.tar')


def train(net, epoch,train_loader, args, criterion, optimizer):
      freq = args.print_freq
      losses = AverageMeter(freq)
      data_time = AverageMeter(freq)
      batch_time = AverageMeter(freq)
      top1 = AverageMeter(freq)
      top5 = AverageMeter(freq)

      net.train()
      end = time.time()

      prefetcher = DataPrefetcher(train_loader)
      input, target = prefetcher.next()
      i = -1      
      
      while input is not None:
          i += 1
          lr = adjust_learning_rate(optimizer, epoch, args)

          data_time.update(time.time() - end)
          output = net(input)

          
    
          if args.loss_type == 'softmax':
              loss = criterion(output, target)
              prec1, prec5 = accuracy(output.data, target, topk=(1,5))
          if args.loss_type == 'a-softmax':
              loss = criterion(output, target)
              prec1, prec5 = accuracy(output[0].data, target, topk=(1,5))
          if args.loss_type == 'regular':
              cos_theta, regular_loss = output
              loss = criterion(cos_theta, target)
              loss = loss + args.reg_weight * regular_loss
              prec1, prec5 = accuracy(cos_theta.data, target, topk=(1,5))

          reduced_loss  = reduce_tensor(loss.data, world_size)
          prec1 = reduce_tensor(prec1, world_size)
          prec5 = reduce_tensor(prec5, world_size)

          losses.update(to_python_float(reduced_loss))
          top1.update(to_python_float(prec1))
          top5.update(to_python_float(prec5))
            
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          torch.cuda.synchronize()

          batch_time.update(time.time() - end)

          end = time.time()
          input, target = prefetcher.next()
              
          if rank == 0 and i % args.print_freq == 0 and i > 1:
              logger = logging.getLogger('global_logger')
              logger.info('Epoch: [{0}]/[{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'LR {lr:.3f}'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
          

  


if __name__ == '__main__':
    main()
