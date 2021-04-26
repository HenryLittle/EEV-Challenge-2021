import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy import signal
from scipy import ndimage
content = pd.read_csv('/data0/EEV/eev-csv/train.csv')
vidmap_path = '/data0/EEV/code/tools/vidmap_train.txt'
vidmap = [x.strip().split(' ') for x in open(vidmap_path)]

from utils import interpolate_output, correlation
import torch

best_conf = (0, 0, 0)

def apply_butter(order, w):
    b, a = signal.butter(order, w, analog=False)
    emp = vid_labels[:, :][::6]
    filt_emp = signal.filtfilt(b, a, emp, axis=0)
    filt_emp = np.maximum(0, filt_emp)
    filt_emp = np.maximum(filt_emp, emp)
    return filt_emp

def test_result(filt_emp):
    global best_conf
    # do a correlation evalutation
    filt_emp = np.concatenate((filt_emp, filt_emp[-1:]))
    output = interpolate_output(torch.from_numpy(filt_emp), 1, 6)
    cor, _ = correlation(output[:vid_labels.shape[0]], torch.   from_numpy(vid_labels))
    print('Correlation:', cor.item())
    return cor.item()
    # if cor.item() > best_conf[0]:
    #     best_conf = (cor.item(), order, w)



vid, st = vidmap[0]
_, ed = vidmap[1]

labels = np.asarray(content.iloc[:,2:], dtype=np.float32)
vid_labels = labels[int(st) : int(ed)]

b, a = signal.butter(1, 0.1, analog=False)
res = []
for i in range(vid_labels.shape[1]):
    emp = vid_labels[:, i][::6]
    # filt_emp = signal.filtfilt(b, a, emp)
    # filt_emp = signal.medfilt(emp)
    filt_emp = ndimage.gaussian_filter1d(emp, 0.8)
    filt_emp = np.maximum(0, filt_emp)
    res.append(filt_emp)
    # filt_emp = np.maximum(filt_emp, emp)
    plt.figure()
    plt.subplot(1,1,1)
    plt.plot(emp, color='silver', label='Original')
    plt.plot(filt_emp, color='#3465a4', label='filtered')
    plt.grid(True, which='both')
    plt.legend(loc='best')
    plt.savefig('img/filter_{}.png'.format(i))
# res = signal.medfilt(vid_labels[:, :][::6], [3, 1])
res = np.stack(res, axis=1)
print(res.shape, vid_labels.shape)
test_result(res)
# test_butter(1, 0.2)

# for o in range(1, 8, 1):
#     for w in np.arange(0.01, 0.41, 0.01):
#         test_butter(o, w)
# print(best_conf)

