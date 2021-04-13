import h5py as h5
import numpy as np

diff = []
img = h5.File('/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5','r')
audio = h5.File('/data0/EEV/vggish_audio_features.hdf5', 'r')

for k in img.keys():
    if k not in audio.keys():
        diff.append(k)

print(diff)
# x = list(set([k for k in audio.keys()]))
# print(len(k), len(audio.keys()))
