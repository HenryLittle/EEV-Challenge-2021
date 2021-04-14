import h5py as h5
import numpy as np
import pandas as pd


print("[1] Check test vidmap length")

vidmap = [int(x.strip().split(' ')[1]) for x in open('/data0/EEV/code/tools/vidmap_test.txt')]
vidmap_sum = np.sum(vidmap)
print(vidmap_sum, vidmap_sum == 2890742)


vidmap_dict = {x.strip().split(' ')[0]: int(x.strip().split(' ')[1]) for x in open('/data0/EEV/code/tools/vidmap_test.txt')}
c1 = pd.read_csv('/data0/EEV/code/baseline/test_output.csv')
v1 = c1['Video ID'].to_list()
keys = set(v1)
res = {}
for k in v1:
    if k in res:
        res[k] += 1
    else:
        res[k] = 1
for k in keys:
    assert vidmap_dict[k] == res[k], '%s %d %d' % (k, vidmap_dict[k], res[k])
