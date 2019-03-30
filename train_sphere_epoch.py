from __future__ import print_function
import sys
import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import distributed
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from memcached_dataset import McDataset
from apex.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from utils import *
import net_sphere_2
from verify_LFW import verify_LFW

parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--base_lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='')
parser.add_argument('--ckpt', default='experiment/softmax', type=str, help='')
parser.add_argument('--data_dir', default='/mnt/lustre/xujingyi/sphereface/preprocess/result/CASIA-WebFace-112X96', type=str)
parser.add_argument('--val_data_dir', default='data/lfw-112X96', type=str)
parser.add_argument('--data_list', default='/mnt/lustre/xujingyi/sphereface/preprocess/result/webface.txt', type=str)
parser.add_argument('--loss_type', default='softmax', type=str)
parser.add_argument('--transforms', default='softmax', type=str)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--top', default=1, type=int)
parser.add_argument('--save_freq', default=1000, type=str)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--resume', default=-1, type=int)
parser.add_argument('--classnum', default=10574, type=int)
parser.add_argument('--distributed', default=False, type=str)
parser.add_argument('--lr_steps', nargs='+', type=int)
parser.add_argument('--lr_mults', default=0.1, type=float)
parser.add_argument('--gamma', default=0.12, type=float)
parser.add_argument('--power', default=1, type=float)
parser.add_argument('--margin', default=0.5, type=float)
parser.add_argument('--LambdaMax', default=1000, type=float)
parser.add_argument('--easy_margin', default=True, type=str)
parser.add_argument('--radius', default=None, type=float)
parser.add_argument('--pretrained', default=None, type=str)
parser.add_argument('--warmup_epochs', default=10, type=int)
parser.add_argument('--reg', default='warmup', type=str)
parser.add_argument('--reg_weight', default=15, type=float)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()



def main():

    global  rank, world_size, softmax_loss_record, reg_loss_record, iters
    if args.distributed:
      rank, world_size = dist_init("1234")
    else:
      rank = 0
      world_size = 0
      
    if rank == 0:
      if not os.path.exists('{}/checkpoints'.format(args.ckpt)):
         os.makedirs('{}/checkpoints'.format(args.ckpt))
      if not os.path.exists('{}/features'.format(args.ckpt)):
         os.makedirs('{}/features'.format(args.ckpt))
      if not os.path.exists('{}/logs'.format(args.ckpt)):
         os.makedirs('{}/logs'.format(args.ckpt))
      if not os.path.exists('{}/plots'.format(args.ckpt)):
         os.makedirs('{}/plots'.format(args.ckpt))

    softmax_loss_record = []
    reg_loss_record = []
    iters = []

    if rank == 0:
       logger = create_logger('global_logger', '{}/logs/{}.txt'.format(args.ckpt,time.time()))
       logger.info('{}'.format(args))
    else:
       logger = None
    
    #net = nn.parallel.distributed.DistributedDataParallel(net)
    
    # build dataset
    data_dir = args.data_dir
    data_list = args.data_list

    train_dataset = McDataset(data_dir, data_list, transforms.Compose([
            transforms.RandomHorizontalFlip()]))

    train_sampler = None

    if args.distributed:
       train_sampler = distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    # create model
    print("=> creating model '{}'".format(args.net))
    net = getattr(net_sphere_2,args.net)(classnum=args.classnum, head=args.loss_type, radius=args.radius, margin=args.margin, easy_margin=args.easy_margin, top=args.top)
    net = net.cuda()

    if args.distributed:
       net = DDP(net)
    
    # build optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
 
    if args.loss_type == 'a-softmax':
        criterion = net_sphere_2.AngleLoss(LambdaMax=args.LambdaMax, gamma=args.gamma, power=args.power).cuda()
    if args.loss_type == 'softmax':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.loss_type == 'norm-regular':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.loss_type == 'arcface':
        criterion = net_sphere_2.ArcMarginLoss(s=args.radius).cuda()

    start_epoch = 0
    # optionally resume from a pretrained model
    if args.resume >= 0:
        model_name = 'epoch_'+str(args.resume)+'_ckpt.pth.tar'
        model_path = os.path.join(args.ckpt, 'checkpoints', model_name)
        checkpoint = torch.load(model_path)
        start_epoch = int(checkpoint['epoch'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.pretrained:
        start_epoch = 0

    ## start training 
    net.train()
    prefetcher = DataPrefetcher(train_loader)
    freq = args.print_freq
    end = time.time()
    
    for epoch in range(start_epoch,args.epochs):
        if args.distributed:
          train_sampler.set_epoch(epoch)
        train(net, epoch, train_loader, args, criterion, optimizer)
        if rank == 0:
             save_checkpoint({
               'epoch': epoch+1,
               'state_dict': net.state_dict(),
               'optimizer': optimizer.state_dict()
               },args,'epoch_'+str(epoch+1)+'_ckpt.pth.tar')

    draw_plot(softmax_loss_record, reg_loss_record, iters, args) 
    save_record(softmax_loss_record, reg_loss_record, args) 




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
              softmax_loss = loss
              prec1, prec5 = accuracy(output.data, target, topk=(1,5))
          if args.loss_type == 'arcface':
              loss = criterion(output, target)
              softmax_loss = loss
              prec1, prec5 = accuracy(output[0].data, target, topk=(1,5))
          if args.loss_type == 'norm-regular':
              cos_theta, regular_loss = output
              softmax_loss = criterion(cos_theta, target)
              regular_weight = adjust_weight(epoch+1, args)
              loss = softmax_loss+regular_loss*regular_weight
              prec1, prec5 = accuracy(cos_theta.data, target, topk=(1,5))
          if args.loss_type == 'a-softmax':
              cos_theta, phi_theta, regular_loss = output
              softmax_loss, lamda = criterion((cos_theta, phi_theta), target)
              regular_weight = adjust_weight(epoch+1, args)
              loss = softmax_loss + regular_loss*regular_weight
              prec1, prec5 = accuracy(cos_theta.data, target, topk=(1,5))
                    

          if args.distributed:
             reduced_loss  = reduce_tensor(loss.data, world_size)
             prec1 = reduce_tensor(prec1, world_size)
             prec5 = reduce_tensor(prec5, world_size)
          else:
             reduced_loss = loss.data
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
              niters = epoch * len(train_loader) + i
              iters.append(niters)
              logger = logging.getLogger('global_logger')
              loss_info = 'Epoch: [{0}]/[{1}/{2}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' \
                      'LR {lr:.4f}'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr)
              softmax_loss_record.append(softmax_loss.data.item())
              if args.loss_type == 'norm-regular' or args.loss_type == 'a-softmax':
                loss_info = loss_info + '\tRegLoss {reg_loss:.3f}' \
                           '\tRegWeight {regweight:.3f}' \
                           '\tSoftmax Loss {softmaxloss:.3f}'.format(reg_loss=regular_loss.item(),regweight=regular_weight, softmaxloss=softmax_loss.data.item())
                reg_loss_record.append(regular_loss.item())
              if args.loss_type == 'a-softmax':
                loss_info = loss_info + '\tLamda {lamda: .3f}'.format(lamda=lamda)

              logger.info(loss_info)
  


if __name__ == '__main__':
    main()
