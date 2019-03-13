from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from dataset import ImageDataset
import net_sphere_2

from apex.parallel import DistributedDataParallel as DDP 

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold



parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--lfw', default='../../dataset/face/lfw/lfw-112X96', type=str)
parser.add_argument('--ckpt', default='experiment/softmax', type=str)
parser.add_argument('--model', default=14000, type=int)
parser.add_argument('--loss_type', default='softmax', type=str)

args = parser.parse_args()

predicts=[]
net = getattr(net_sphere_2,args.net)(classnum=10575, head=args.loss_type)
model_name = 'epoch_'+str(args.model)+'_ckpt.pth.tar'
model_path = os.path.join(args.ckpt, 'checkpoints', model_name)
#checkpoint = torch.load(model_path)['state_dict']
net.cuda()
#from collections import OrderedDict
#new_state_dict = OrderedDict()
#for k, v in checkpoint.items():
#  name = k[7:]
#  new_state_dict[name] = v
#net.load_state_dict(new_state_dict)
net.load_state_dict(torch.load(model_path))
net.eval()
net.feature = True



with open('data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]

for i in range(6000):
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
    
    img1 = cv2.imread(os.path.join(args.lfw, name1))
    img2 = cv2.imread(os.path.join(args.lfw, name2))
    imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    for j in range(len(imglist)):
        imglist[j] = imglist[j].transpose(2,0,1).reshape((1,3,112,96))
        imglist[j][0][0] -= 0.485*255
        imglist[j][0][1] -= 0.456*255
        imglist[j][0][2] -= 0.406*255
        imglist[j][0][0] /= 0.229*255
        imglist[j][0][1] /= 0.224*255
        imglist[j][0][2] /= 0.225*255
    print(imglist[0][0][0][:3][:3])   
    img = np.vstack(imglist)
    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    output = net(img)
    f = output.data
    f1,f2 = f[0],f[2]
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    print('Calculating pair {} in total pairs {}'.format(i, 6000))
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))
   

print("Calculating similarity score is done!")
accuracy = []
thd = []
folds = KFold(n=6000, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
cos_predicts = []
for line in predicts:
    line = line.strip().split()
    cos_predicts.append(line)
cos_predicts = np.array(cos_predicts)
print(cos_predicts.shape)
for idx, (train, test) in enumerate(folds):
    print("Finding threshold for the {} th fold".format(idx))
    best_thresh = find_best_threshold(thresholds, cos_predicts[train])
    accuracy.append(eval_acc(best_thresh, cos_predicts[test]))
    thd.append(best_thresh)
if not os.path.exists('{}/results'.format(args.ckpt)):
    os.makedirs('{}/results'.format(args.ckpt))
result_file = open('{}/results/performance.txt'.format(args.ckpt), 'a')
line = 'epoch={} LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(args.model, np.mean(accuracy), np.std(accuracy), np.mean(thd))
result_file.write(line + '\n')
