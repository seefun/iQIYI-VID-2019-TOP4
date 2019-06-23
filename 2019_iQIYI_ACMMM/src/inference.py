# libraries
import numpy as np
import os
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time 
import sys
import math
import gc
import scipy

cudnn.benchmark = True

if sys.version_info >= (3, ):
    import pickle
    def load_pickle(fin):
        return pickle.load(fin, encoding='bytes')
else:
    import cPickle as pickle
    def load_pickle(fin):
        return pickle.load(fin)

f_handler=open('/data/logs/out.log','w')
sys.stdout=f_handler
sys.stderr=f_handler

#param:
#TEST_PICKLE = '../data/face_val_v2.pickle'

MODEL1_PATH1 = '/project/save/2-layer-sigmoid-model0.pt'
MODEL1_PATH2 = '/project/save/2-layer-sigmoid-model1.pt'
MODEL1_PATH3 = '/project/save/2-layer-sigmoid-model2.pt'
MODEL1_PATH4 = '/project/save/2-layer-sigmoid-model3.pt'
MODEL1_PATH5 = '/project/save/2-layer-sigmoid-model4.pt'

MODEL2_PATH1 = '/project/save/3-sigmoid-model0.pt'
MODEL2_PATH2 = '/project/save/3-sigmoid-model1.pt'
MODEL2_PATH3 = '/project/save/3-sigmoid-model2.pt'
MODEL2_PATH4 = '/project/save/3-sigmoid-model3.pt'
MODEL2_PATH5 = '/project/save/3-sigmoid-model4.pt'

MODEL3_PATH1 = '/project/save/2-swish-model0.pt'
MODEL3_PATH2 = '/project/save/2-swish-model1.pt'
MODEL3_PATH3 = '/project/save/2-swish-model2.pt'
MODEL3_PATH4 = '/project/save/2-swish-model3.pt'
MODEL3_PATH5 = '/project/save/2-swish-model4.pt'

MODEL4_PATH1 = '/project/save/3-swish-model0.pt'
MODEL4_PATH2 = '/project/save/3-swish-model1.pt'
MODEL4_PATH3 = '/project/save/3-swish-model2.pt'
MODEL4_PATH4 = '/project/save/3-swish-model3.pt'
MODEL4_PATH5 = '/project/save/3-swish-model4.pt'

MODEL5_PATH1 = '/project/save/3-swish-new-model0.pt'
MODEL5_PATH2 = '/project/save/3-swish-new-model1.pt'
MODEL5_PATH3 = '/project/save/3-swish-new-model2.pt'
MODEL5_PATH4 = '/project/save/3-swish-new-model3.pt'
MODEL5_PATH5 = '/project/save/3-swish-new-model4.pt'


TEST_PICKLE = '/data/materials/feat/face_test.pickle'
OUTPUT_PATH = '/data/result/result.txt'
LOG_DIR = '/data/logs/'

try:
    os.mkdir('/data/result')
except:
    pass

 
# data  (face val : weighted average)
start = time.time()
face_path_val = TEST_PICKLE
print('loading {}...'.format(face_path_val))
with open(face_path_val, 'rb') as fin:
    face_feats_dict_val = load_pickle(fin)
end = time.time()
print('loaded {}'.format(face_path_val)) 
print('Load Data Costs: %.2f s'%(end-start))


    
new_val_dict = {}

for video_ind, video_name in enumerate(face_feats_dict_val):
    all_weight = 0
    weighted_feats = [0]*512
    face_feats = face_feats_dict_val[video_name]
    for ind, face_feat in enumerate(face_feats):
        [frame_str, bbox, det_score, quality_score, feat] = face_feat
        [x1, y1, x2, y2] = bbox
        if quality_score<0:
            quality_score*=0.0
        elif quality_score<20:
            quality_score*=0.2
        elif quality_score<30:
            quality_score*=0.3
        elif quality_score<40:
            quality_score*=0.6
        else:
            pass
        weight = quality_score*det_score
        all_weight += weight
        weighted_feats += feat*weight
    weighted_feats = np.array(weighted_feats, dtype='float32')/all_weight
    new_val_dict[video_name.decode(encoding="utf-8")] = weighted_feats
        
del face_feats_dict_val
gc.collect()

name_list_val = []
feature_list_val = []

for video_ind, video_name in enumerate(new_val_dict):
    face_feat = new_val_dict[video_name]
    name_list_val.append(video_name)
    feature_list_val.append(np.array(face_feat, dtype='float32'))

del new_val_dict
gc.collect()

# model
def get_label(attribute_ids):
    one_hot = torch.zeros(10035).scatter_(0, torch.LongTensor([attribute_ids]), 1)
    return one_hot
    
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
        if self.is_train:
            label = self.train_class_dict[name]
        else:
            label = 0
        label = get_label(label) #one_hot_label
        return feats,label
        
test_set = faceDataset_test(name_list_val, feature_list_val, is_train=False)
batch_size = 512
num_workers = 0
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

from scipy.special import expit as sigmoid

def get_predict(name): 
    
    saved_dict = torch.load(name)
    model_conv.load_state_dict(saved_dict)
    
    preds = []
    with torch.no_grad():
        for batch_i, (data, target) in enumerate(test_loader):
            data = data.cuda()
            output = model_conv(data).detach()
            output = output.cpu().numpy()
            for pred in output:
                score = sigmoid(pred)
                preds.append(score)
    preds = np.nan_to_num(np.array(preds))
    return preds
    
            
def write_result(preds, path):
    with open(path, 'w') as outf:
        for i in range(1,10035):
            outf.write(str(i))
            scores = []
            for pred in preds:
                scores.append(pred[i])
            scores = np.array(scores)
            for _,index in enumerate(np.argsort(-scores)):
                if _==100:
                    break
                outf.write(' '+name_list_val[index]+'.mp4')
            outf.write('\n')

#model
start_t = time.time()

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x*self.sigmoid(x)

#2-layer-sigmoid
model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True), nn.Sigmoid(), nn.BatchNorm1d(1024),
                                      nn.Dropout(0.25),  nn.Linear(in_features=1024, out_features=10035, bias=True))

model_conv.cuda()
model_conv.eval()
preds = get_predict(MODEL1_PATH1) 
preds = preds + get_predict(MODEL1_PATH2)
preds = preds + get_predict(MODEL1_PATH3)
preds = preds + get_predict(MODEL1_PATH4)
preds = preds + get_predict(MODEL1_PATH5)
print('model 1 inference done')

#3-layer-sigmoid
model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True), nn.Sigmoid(), nn.BatchNorm1d(512), nn.Dropout(0.2), 
                           nn.Linear(in_features=512, out_features=1024, bias=True), nn.Sigmoid(), nn.BatchNorm1d(1024), nn.Dropout(0.5), 
                           nn.Linear(in_features=1024, out_features=10035, bias=True))

model_conv.cuda()
model_conv.eval()
preds = preds + get_predict(MODEL2_PATH1)
preds = preds + get_predict(MODEL2_PATH2)
preds = preds + get_predict(MODEL2_PATH3) 
preds = preds + get_predict(MODEL2_PATH4)
preds = preds + get_predict(MODEL2_PATH5)
print('model 2 inference done')

#2-layer-swish
model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True), swish(), nn.BatchNorm1d(1024),
                           nn.Dropout(0.25),  nn.Linear(in_features=1024, out_features=10035, bias=True))
                           
model_conv.cuda()
model_conv.eval()
preds = preds + get_predict(MODEL3_PATH1) 
preds = preds + get_predict(MODEL3_PATH2)
preds = preds + get_predict(MODEL3_PATH3)
preds = preds + get_predict(MODEL3_PATH4)
preds = preds + get_predict(MODEL3_PATH5)
print('model 3 inference done')

#3-layer-swish
model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=768, bias=True), swish(), nn.Dropout(0.2),
                           nn.Linear(in_features=768, out_features=1024, bias=True), swish(), nn.Dropout(0.5), 
                           nn.Linear(in_features=1024, out_features=10035, bias=True))

model_conv.cuda()
model_conv.eval()
preds = preds + get_predict(MODEL4_PATH1)
preds = preds + get_predict(MODEL4_PATH2)
preds = preds + get_predict(MODEL4_PATH3)
preds = preds + get_predict(MODEL4_PATH4)
preds = preds + get_predict(MODEL4_PATH5)
print('model 4 inference done')

#3-layer-swish-new
model_conv = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True), swish(), nn.Dropout(0.5), 
                          nn.Linear(in_features=1024, out_features=1024, bias=True), swish(), nn.Dropout(0.5), 
                          nn.Linear(in_features=1024, out_features=10035, bias=True))

model_conv.cuda()
model_conv.eval()
preds = preds + get_predict(MODEL5_PATH1)
preds = preds + get_predict(MODEL5_PATH2)
preds = preds + get_predict(MODEL5_PATH3)
preds = preds + get_predict(MODEL5_PATH4)
preds = preds + get_predict(MODEL5_PATH5)
print('model 5 inference done')


end_t = time.time()

print('Inference time: %.2f s' % (end_t - start_t))

write_result(preds, OUTPUT_PATH)

end_all = time.time()
print('Write result cost: %.2f s'% (end_all - end_t))

print('Done')


#from utils.evaluation_map import calculate_map
#gt_val_path = '../data/val_gt.txt'
#my_val_path = 'my_val.txt'
#mAP = calculate_map(gt_val_path, my_val_path)
#print('[mAP]:%.6f'%mAP)
