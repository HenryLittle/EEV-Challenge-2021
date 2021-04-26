import xgboost as xgb
import pandas as pd
import numpy as np
import h5py
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from utils import correlation, interpolate_output
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

train_img_feat_path = '/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5'
train_csv_path = '/data0/EEV/eev-csv/train.csv'
train_vidmap_path = '/data0/EEV/code/tools/vidmap_train.txt'

val_img_feat_path = '/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5'
val_csv_path = '/data0/EEV/eev-csv/val.csv'
val_vidmap_path = '/data0/EEV/code/tools/vidmap_val.txt'

def compose_data(img_feat_path, csv_path, vidmap_path, step={'data':6, 'target':6}, cat=True):
    
    data = []
    target = []
    img_feat = h5py.File(img_feat_path, 'r') # {key:[x, 2048]} 6Hz
    vidmap_list = {x.strip().split(' ')[0]:int(x.strip().split(' ')[1]) for x in open(vidmap_path) if x.strip().split(' ')[0] in img_feat.keys()} # {vid:st_idx}
    labels = pd.read_csv(csv_path)
    labels = np.asarray(labels.iloc[:,(2 + 3)])
    print('feats:', len(img_feat.keys()), 'vidmap:', len(vidmap_list.keys()))
    for k in tqdm(vidmap_list.keys()):
        vid_len = img_feat[k].shape[0]
        vid_st_idx = vidmap_list[k]
        indices = [x for x in range(0, vid_len, step['data'])]
        vid_indices = [min(x + vid_st_idx, len(labels) - 1) for x in range(0, vid_len, step['target'])]

        data.append(img_feat[k][indices])
        target.append(labels[vid_indices])
        # print(data[0].shape, target[0].shape)
    if cat:
        data = np.concatenate(data, axis=0)
        target = np.concatenate(target, axis=0)
    return data, target


def load_dataset():
    print('loading train ...')
    train, train_target = compose_data(train_img_feat_path, train_csv_path, train_vidmap_path)
    print('loading validation ...')
    val, val_target = compose_data(val_img_feat_path, val_csv_path, val_vidmap_path, {'data':6, 'target':1}, False)
    return train, train_target, val, val_target

def main():
    # compose data for xgboost
    train, train_target, val, val_target = load_dataset() # np arrays [X 2048] [X 15] (large X)
    # train = np.array([[1,2,3,4,5], [2,3,4,5,6], [4,5,6,7,8], [5,6,7,8,9]])
    # train_target = np.array([3,4,5,6])
    print(train.shape, train_target.shape, val[0].shape, val_target[0].shape)
    regressor = xgb.XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor')
    # 
    regressor.fit(train, train_target)
    score = regressor.score(train, train_target)
    print('Score:', score)
    # print(regressor.predict(np.array([[6,7,8,9,10]])))
    video_count = len(val)
    corr_mean = 0.0
    for i in range(video_count):
        preds = regressor.predict(val[i])
        preds = torch.from_numpy(preds).cuda()
        preds = torch.cat([preds, preds[-1:]])
        preds = preds.unsqueeze(1) # T -> T 1
        preds = interpolate_output(preds, 1 , 6)
        pred_len = val_target[i].shape[0]
        preds = preds[:pred_len]
        val_ti = torch.from_numpy(val_target[i]).cuda()
        val_ti = val_ti.unsqueeze(1) # T -> T 1
        corr, _ = correlation(preds, val_ti)
        corr_mean += corr.item()
    print('Correlation:', corr_mean/video_count)


if __name__ == '__main__':
    main()