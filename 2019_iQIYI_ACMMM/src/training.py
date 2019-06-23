# libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time 
import tqdm
import random
import pdb
import sys
import math
import gc
from PIL import Image
from collections import defaultdict
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

cudnn.benchmark = True

if sys.version_info >= (3, ):
    import pickle
    def load_pickle(fin):
        return pickle.load(fin, encoding='bytes')
else:
    import cPickle as pickle
    def load_pickle(fin):
        return pickle.load(fin)
    
from evaluation_map import calculate_map

SEED = 2333
base_dir = '../input/'
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)

# #TRAIN_SET

# face train
face_path_train = '../data/face_train_v2.pickle'
print('loading {}...'.format(face_path_train))
with open(face_path_train, 'rb') as fin:
    face_feats_dict_train = load_pickle(fin)
    
# face val
face_path_val = '../data/face_val_v2.pickle'
print('loading {}...'.format(face_path_val))
with open(face_path_val, 'rb') as fin:
    face_feats_dict_val = load_pickle(fin)
    
face_feats_dict_train.update(face_feats_dict_val)

# trainval gt
train_class_dict = {}
with open('../data/trainval_gt.txt') as train_gt:
    for line in train_gt.readlines():
        name = line.split()[0].split('.')[0]
        label = int(line.split()[1])
        train_class_dict[name] = label
        
# feature 
name_list = []
feature_list = []
det_score_lists = []
quality_score_lists = []

for video_ind, video_name in enumerate(face_feats_dict_train):
    face_feats_list = []
    det_score_list = []
    quality_score_list = []
    face_feats = face_feats_dict_train[video_name]
    for ind, face_feat in enumerate(face_feats):
        [frame_str, bbox, det_score, quality_score, feat] = face_feat
        [x1, y1, x2, y2] = bbox
        if det_score >= 0.7 and quality_score>=25:
            face_feats_list.append(np.array(feat, dtype='float32'))
            det_score_list.append(det_score)
            quality_score_list.append(quality_score)
    if len(face_feats_list)>0:
        name_list.append(video_name.decode(encoding="utf-8"))
        feature_list.append(face_feats_list)
        det_score_lists.append(det_score_list)
        quality_score_lists.append(quality_score_list)
        
del face_feats_dict_train
del face_feats_dict_val
gc.collect()

y = []
for name in name_list:
    y.append(train_class_dict[name])
    
sfolder = StratifiedKFold(n_splits=5,random_state=SEED,shuffle=True)
kfold = []

for train, test in sfolder.split(name_list,y):
    kfold.append((train, test))
    print('Train: %s | test: %s' % (train, test))
    print(" ")

#### change training fold #####
FOLD = 0 #0-4
tr, val = kfold[FOLD]

def get_label(attribute_ids):
    one_hot = torch.zeros(10035).scatter_(0, torch.LongTensor([attribute_ids]), 1)
    return one_hot
    
class faceDataset_train(Dataset):
    def __init__(self, name_list, feature_list, train_class_dict={}, is_train=True):
        self.name_list = name_list
        self.feature_list = feature_list
        self.train_class_dict = train_class_dict
        self.is_train = is_train

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        feats = self.feature_list[idx]
        det_scores = det_score_lists[idx]
        quality_scores = quality_score_lists[idx]
        
        rand1 = random.randint(0,len(feats)-1)
        rand2 = random.randint(0,len(feats)-1)
        rand3 = random.randint(0,len(feats)-1)
        rand4 = random.randint(0,len(feats)-1)
        rand5 = random.randint(0,len(feats)-1)
        feats1 = feats[rand1]
        feats2 = feats[rand2]
        feats3 = feats[rand3]
        feats4 = feats[rand4]
        feats5 = feats[rand5]
        score1 = quality_scores[rand1]*det_scores[rand1]
        score2 = quality_scores[rand2]*det_scores[rand2]
        score3 = quality_scores[rand3]*det_scores[rand3]
        score4 = quality_scores[rand4]*det_scores[rand4]
        score5 = quality_scores[rand5]*det_scores[rand5]
        feats_3 = (feats1*score1 + feats2*score2 + feats3*score3) / (score1 +score2 + score3)
        feats_5 = (feats1*score1 + feats2*score2 + feats3*score3 + feats4*score4 + feats5*score5) / (score1 + score2 + score3 + score4 + score5)
        choise = random.random()
        if choise<0.25: # random choose raw data
            feats = feats1
        elif choise<0.9:
            feats = feats_3
        else:
            feats = feats_5
        
        if self.is_train:
            label = self.train_class_dict[name]
        else:
            label = 0
        label = get_label(label) #one_hot_label
        return feats,label

class faceDataset_test(Dataset):
    def __init__(self, name_list, feature_list, train_class_dict={}, is_train=True):
        self.name_list = name_list
        self.feature_list = feature_list
        self.train_class_dict = train_class_dict
        self.is_train = is_train

    def __len__(self):
        return len(self.name_list) 

    def __getitem__(self, idx):
        name = self.name_list[idx]
        feats = self.feature_list[idx]
        det_scores = det_score_lists[idx]
        quality_scores = quality_score_lists[idx]
        
        all_weight = 0
        weighted_feats = [0]*512
        for ind, face_feat in enumerate(feats):
            quality_s = quality_scores[ind]
            det_s = det_scores[ind]
            if quality_s<40:
                quality_s*=0.6
            else:
                pass
            weight = quality_s*det_s
            all_weight += weight
            weighted_feats += face_feat*weight
        weighted_feats = np.array(weighted_feats, dtype='float32')/all_weight
        
        if self.is_train:
            label = self.train_class_dict[name]
        else:
            label = 0
        label = get_label(label) #one_hot_label
        return weighted_feats, label
        
dataset = faceDataset_train(name_list, feature_list, train_class_dict)
test_set = faceDataset_test(name_list, feature_list, train_class_dict, is_train=True)
batch_size = 512
num_workers = 0

train_sampler = SubsetRandomSampler(list(tr)) 
valid_sampler = SubsetRandomSampler(list(val))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x*self.sigmoid(x)

#2-layer-sigmoid
#model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True), nn.Sigmoid(), nn.BatchNorm1d(1024),
#                                      nn.Dropout(0.25),  nn.Linear(in_features=1024, out_features=10035, bias=True))

#3-layer-sigmoid
model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True), nn.Sigmoid(), nn.BatchNorm1d(512), nn.Dropout(0.2), 
                           nn.Linear(in_features=512, out_features=1024, bias=True), nn.Sigmoid(), nn.BatchNorm1d(1024), nn.Dropout(0.5), 
                           nn.Linear(in_features=1024, out_features=10035, bias=True))

#2-layer-swish
#model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True), swish(), nn.BatchNorm1d(1024),
#                                      nn.Dropout(0.25),  nn.Linear(in_features=1024, out_features=10035, bias=True))
    
#3-layer-swish-new
# model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=768, bias=False), nn.BatchNorm1d(768), swish(), nn.Dropout(0.2),
#                            nn.Linear(in_features=768, out_features=1024, bias=False), nn.BatchNorm1d(1024), swish(), nn.Dropout(0.5),
#                            nn.Linear(in_features=1024, out_features=10035, bias=True)
#                           )

#3-layer-swish
#model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=768, bias=True), swish(), nn.Dropout(0.2), 
#                          nn.Linear(in_features=768, out_features=1024, bias=True), swish(), nn.Dropout(0.5), 
#                          nn.Linear(in_features=1024, out_features=10035, bias=True))

#3-layer-swish-new
#model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True), swish(), nn.Dropout(0.5), 
#                          nn.Linear(in_features=1024, out_features=1024, bias=True), swish(), nn.Dropout(0.5), 
#                          nn.Linear(in_features=1024, out_features=10035, bias=True))

for m in model_conv:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        #nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()
    
class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.softmax_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, logits, labels):
        loss_softmax = self.softmax_loss(logits, torch.argmax(labels, dim=1))
        loss_focal = self.focal_loss(logits, labels)
        #print('focal:%.6f'%loss_focal)
        #print('softmax:%.6f'%loss_softmax)
        return 0.02 * loss_softmax + 0.98 * loss_focal
    
class StartLoss(nn.Module):
    def __init__(self):
        super(StartLoss, self).__init__()
        self.softmax_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, logits, labels):
        loss_softmax = self.softmax_loss(logits, torch.argmax(labels, dim=1))
        loss_focal = self.focal_loss(logits, labels)
        #print('focal:%.6f'%loss_focal)
        #print('softmax:%.6f'%loss_softmax)
        return 0.25 * loss_softmax + 0.75 * loss_focal

model_conv.cuda()

criterion = StartLoss()

optimizer = optim.Adam(model_conv.parameters(), lr=0.00002)

scheduler = StepLR(optimizer, 50, gamma=0.5)

n_epochs = 500
best_acc = 0
for epoch in range(1, n_epochs+1):
    #scheduler.step()
    print(time.ctime(), 'Epoch:', epoch)

    train_loss = []
    train_auc = []  
    
    if epoch==25:   
        criterion = FocalLoss()
    if epoch==10:
        optimizer = optim.Adam(model_conv.parameters(), lr=0.00015)
        scheduler = StepLR(optimizer, 50, gamma=0.5)
    if epoch==100:
        criterion = CombineLoss()        
        
    ### Train    
    for tr_batch_i, (data, target) in enumerate(train_loader):
        
        model_conv.train()

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model_conv(data)
        
        loss = criterion(output, target)
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        
        
        if (tr_batch_i+1)%150 == 0:   
            print('train_loss:%.6f'%np.mean(train_loss))
            train_loss = []
            
            a = target.data.cpu().numpy()
            b = output.detach().cpu().numpy()
            pred = []
            gt = []
            for i in a:
                gt.append(np.argmax(i))
            for i in b:
                pred.append(np.argmax(i))
            correct = 0
            for i in range(len(a)):
                if pred[i]==gt[i]:
                    correct += 1
            acc = correct/len(a)
            print('Train accuracy:%.3f'%acc)
            
    ## Eval    
    model_conv.eval()
    pred = []
    gt = []
    for val_batch_i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()

        output = model_conv(data)
        
        a = target.data.cpu().numpy()
        b = output.detach().cpu().numpy()
        
        for i in a:
            gt.append(np.argmax(i))
        for i in b:
            pred.append(np.argmax(i))
    correct = 0
    for i in range(len(pred)):
        if pred[i]==gt[i]:
            correct += 1
    acc = correct/len(pred)
    print('Val accuracy:%.5f'%acc)

    if acc > best_acc:    
        torch.save(model_conv.state_dict(), 'save/model%d.pt'%FOLD)
        print('Saving model[epoch%d]...'%(epoch))
        best_acc = acc
          
    scheduler.step()
