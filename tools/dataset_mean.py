import pandas as pd
import numpy as np
import pickle as pkl

train = pd.read_csv('/data0/EEV/eev-csv/train.csv')

data = np.asarray(train.iloc[:, 2:])
mean = np.mean(data, axis=0)
print(mean)
std = np.std(data, axis=0)
print(std)

save_dict = {}
save_dict['mean'] = mean
save_dict['std'] = std
with open('mean_std.pkl','wb') as f:
    pkl.dump(save_dict,f)
