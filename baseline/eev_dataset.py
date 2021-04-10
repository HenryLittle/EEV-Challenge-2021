import torch.utils.data as data
import pandas as pd
import numpy as np
import h5py

class EEV_Dataset(data.Dataset):
    def __init__(self, csv_path, vidmap_path, image_feat_path, audio_feat_path, image_freq=2, sample_length=60):
        assert image_freq in [2] # Hz
        self.freq = image_freq # samples per second for images (default is 2Hz)
        self.sample_length = sample_length # 60 total frames 
        self.csv_content = pd.read_csv(csv_path)
        # csv data
        self.emotions = np.asarray(self.csv_content.iloc[:,2:], dtype=np.float32)
        assert len(self.emotions[0]) == 15
        self.vidmap_list = [x.strip().split(' ') for x in open(vidmap_path)]
        # features
        self.image_features = h5py.File(image_feat_path, 'r') # {vid: [x, 2048]} 2Hz
        self.audio_features = h5py.File(audio_feat_path, 'r') # {vid: [x, 128]} 0.96s per sample

        # filter bad samples
        print('vidmap list length:', len(self.vidmap_list))
        self.vidmap_list = [x for x in self.vidmap_list if x[0] in self.image_features.keys()]
        print('vidmap list filter length:', len(self.vidmap_list))

 

    def __len__(self):
        # return total video count
        return len(self.vidmap_list)
    
    def __getitem__(self, index):
        # return a 60 seconds sequence inside a video
        vid, start_idx = self.vidmap_list[index]
        vid_start_idx = int(start_idx)
        vid_end_idx = int(self.vidmap_list[index + 1][1]) if index + 1 < len(self.vidmap_list) else len(self.emotions)

        img_feat = np.asarray(self.image_features[vid])
        au_feat = np.asarray(self.audio_features[vid])
        img_feat_len = img_feat.shape[0]
        au_feat_len = au_feat.shape[0]
        # image, audio and lables shares the same start index (seconds)
        feat_start_idx = self.gen_start_idx(img_feat_len, self.freq)
        # print(img_feat_len, au_feat_len, feat_start_idx)
        img_feat = img_feat[self._sample_indices(feat_start_idx, img_feat_len, self.freq)] # [60, 2048]
        au_feat = au_feat[self._sample_indices(feat_start_idx, au_feat_len, 1)] # [60, 128]
        # load lables
        labels = self.load_lables(vid_start_idx + feat_start_idx * 6, vid_end_idx) # [60, 15]
        return img_feat, au_feat, labels

    
    def gen_start_idx(self, frame_count, freq):
        start_pos = max(0, frame_count // freq - self.sample_length) # start pos in seconds
        return np.random.randint(0, start_pos + 1) # rand[)


    def _sample_indices(self, start_idx, frame_count, freq):
        # repeat last frame
        indices = [(start_idx + x * freq) if (start_idx + x * self.freq) < frame_count else (frame_count - freq) for x in range(self.sample_length)]
        return indices


    def load_lables(self, vid_start_idx, vid_end_idx):
        assert vid_start_idx < vid_end_idx
        indices = self._sample_indices(vid_start_idx, vid_end_idx - vid_start_idx, 6) 
        return self.emotions[indices]

    



