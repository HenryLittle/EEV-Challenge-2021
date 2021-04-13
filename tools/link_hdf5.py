import h5py as h5
import numpy as np
whole = h5.File('/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5','w')
p1 = h5.File('/data0/EEV/feats/EEV_InceptionV3_6Hz_1.hdf5','r')
p2 = h5.File('/data0/EEV/feats/EEV_InceptionV3_6Hz_2.hdf5','r')
p3 = h5.File('/data0/EEV/feats/EEV_InceptionV3_6Hz_3.hdf5','r')
p4 = h5.File('/data0/EEV/feats/EEV_InceptionV3_6Hz_4.hdf5','r')

parts = [p1, p2, p3, p4]

for part in parts:
    for key in part.keys():
        if key not in whole.keys():
            whole[key] = np.asarray(part[key])
print(len(whole.keys()))

whole.close()
p1.close()
p2.close()
p3.close()
p4.close()