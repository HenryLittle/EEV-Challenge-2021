import torch.utils.data as data
import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import os
import pickle as pkl
from scipy import signal, ndimage
from einops import rearrange
from tqdm import tqdm

class EEV_Dataset(data.Dataset):
    def __init__(self, csv_path, vidmap_path, image_feat_path, audio_feat_path, mode='train', lpfilter=None, image_freq=6, sample_length=60, train_freq=1, val_freq=6, test_freq=6, cls_indices=None, repeat_sample=False):
        assert image_freq in [1, 6] # Hz
        assert mode in ['train', 'val', 'test', 'merge']
        self.mode = mode
        self.cls_indices = cls_indices
        self.filter = lpfilter
        self.repeat_sample = repeat_sample
        self.image_freq = image_freq # the intrinsic sample rate of image features
        self.train_freq = train_freq # freqency we trained the model on
        self.test_freq = test_freq   # freqency to test/validate on, if greater than train_freq interpolate is engaged
        self.val_freq = val_freq
        self.sample_length = sample_length # 60 total frames

        if csv_path == None and mode != 'test':
            raise RuntimeError('Empty labels!!')
                    
        if self.mode == 'merge':
            self.vidmap_list = {}
            assert len(vidmap_path) == 2, 'Expected 2 vidmaps got {}'.format(len(vidmap_path))
            for i in range(2):
                print(vidmap_path[i])
                self.vidmap_list[i] = [x.strip().split(' ') for x in open(vidmap_path[i])]
                self.vidmap_list[i] = [x + [i] for x in self.vidmap_list[i]]
            # vid, start_idx, map_idx
            self.vidmap_list = self.vidmap_list[0] + self.vidmap_list[1]
            print('Merged:', len(self.vidmap_list))
            # load csv data
            temp_content = {}
            assert len(csv_path) == 2, 'Expected 2 csv files got {}'.format(len(csv_path))
            for i in range(2):
                temp_content[i] = pd.read_csv(csv_path[i])
                temp_content[i] = np.asarray(temp_content[i].iloc[:,2:], dtype=np.float32)
            self.emotions = temp_content

        else:
            self.vidmap_list = [x.strip().split(' ') for x in open(vidmap_path)]

            # load csv data
            if self.mode != 'test':
                self.csv_content = pd.read_csv(csv_path)
                self.emotions = np.asarray(self.csv_content.iloc[:,2:], dtype=np.float32)
                assert len(self.emotions[0]) == 15

        # features
        im_ext = os.path.splitext(image_feat_path)[-1]
        if im_ext == '.hd5f' or im_ext == '.hdf5':
            self.image_features = h5py.File(image_feat_path, 'r') # {vid: [x, 2048]} 2Hz
        elif im_ext == '.pkl':
            f = open(image_feat_path, 'rb')
            self.image_features = pkl.load(f)
        au_ext = os.path.splitext(audio_feat_path)[-1]
        if au_ext == '.hdf5' or au_ext == '.hd5f':
            self.audio_features = h5py.File(audio_feat_path, 'r') # {vid: [x, 128]} 0.96s per sample
        else:
            self.audio_features = {}
            filenames = os.listdir(audio_feat_path)
            print('Loading audio features from:', audio_feat_path)
            for file in tqdm(filenames):
                f = open(os.path.join(audio_feat_path, file), 'rb')
                p = pkl.load(f)
                f_name = os.path.splitext(file)[0] # filename => vid
                self.audio_features[f_name] = p
                f.close()




        # filter bad samples
        print('vidmap list length:', len(self.vidmap_list))
        self.vidmap_list = [x for x in self.vidmap_list if x[0] in self.image_features.keys()]
        print('vidmap list filtered length:', len(self.vidmap_list))

 

    def __len__(self):
        # return total video count
        return len(self.vidmap_list)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.get_train_item(index)
        elif self.mode == 'val':
            return self.get_val_item(index)
        elif self.mode == 'test':
            return self.get_test_item(index)
        elif self.mode == 'merge':
            return self.get_tv_item(index)
    
    def get_video_info(self, index):
        # return video id and [st,ed) index in lables
        # frame_count = ed - st (6Hz)
        vid, start_idx = self.vidmap_list[index]
        vid_start_idx = int(start_idx)
        vid_end_idx = int(self.vidmap_list[index + 1][1]) if index + 1 < len(self.vidmap_list) else len(self.emotions)
        return vid, vid_start_idx, vid_end_idx
    
    def get_tv_item(self, index):
        vid, start_idx, data_split = self.vidmap_list[index]
        vid_start_idx = int(start_idx)
        if data_split == 0:
            vid_end_idx = int(self.vidmap_list[index + 1][1]) if data_split == self.vidmap_list[index + 1][2] else len(self.emotions[data_split])
        else:
            vid_end_idx = int(self.vidmap_list[index + 1][1]) if index + 1 < len(self.vidmap_list) else len(self.emotions[data_split])
        # load features
        img_feat = np.asarray(self.image_features[vid])
        au_feat = np.asarray(self.audio_features[vid])
        # gen start sec
        feat_start_sec = self.gen_start_idx(img_feat.shape[0], self.image_freq)
        img_feat, au_feat, labels = self.sample_item(feat_start_sec, img_feat, au_feat, self.emotions[data_split], vid_start_idx, vid_end_idx, self.train_freq)
        return img_feat, au_feat, labels

    def get_test_item(self, index):
        vid, total_frames = self.vidmap_list[index] # checked

        img_feat = np.asarray(self.image_features[vid])
        au_feat = np.asarray(self.audio_features[vid])

        img_feat_list = []
        au_feat_list = []

        start_sec = 0

        frame_count = int(total_frames) # total frames (feature freqency considered)
        video_length = frame_count // 6
        if frame_count % 6 != 0:
            video_length += 1
        while start_sec < video_length:
            if self.repeat_sample:
                # Clip R S C
                raise RuntimeError('Nooooo!')
            else:
                img_feat_out = img_feat[self._sample_indices_adv(self.sample_length, (start_sec * self.image_freq), img_feat.shape[0], self.image_freq, self.train_freq)] # [60, 2048]
            au_feat_out = au_feat[self._sample_indices_adv(self.sample_length, start_sec, au_feat.shape[0], 1, self.train_freq)] # [60, 128]
            img_feat_list.append(img_feat_out)
            au_feat_list.append(au_feat_out)
            start_sec += (self.sample_length // self.train_freq) # step the correspond time step
        assert len(img_feat_list) == len(au_feat_list)
        return img_feat_list, au_feat_list, frame_count, vid


    def get_val_item(self, index):
        # return the full sequence in 60 frames segments
        vid, vid_start_idx, vid_end_idx = self.get_video_info(index)

        img_feat = np.asarray(self.image_features[vid])
        au_feat = np.asarray(self.audio_features[vid])

        img_feat_list = []
        au_feat_list = []
        labels_list = []

        start_sec = 0 # assumed to be second in video
        frame_count = vid_end_idx - vid_start_idx # 6 Hz frame count
        video_length = frame_count // 6 # total frames in video
        if frame_count % 6 != 0:
            video_length += 1
        while start_sec < video_length:
            # print(self._sample_indices_adv(self.sample_length, start_sec, au_feat.shape[0], 1, self.train_freq))
            if self.repeat_sample:
                # Clip R S C
                repeat = int(self.test_freq // self.train_freq)
                img_r = []
                
                    # print('get val offset:', offset)
                img_feat_out = img_feat[self._sample_indices_adv(self.sample_length * repeat, (start_sec * self.image_freq), img_feat.shape[0], self.image_freq, self.train_freq)]
                print(img_feat.shape)
                img_feat_out = rearrange(img_feat_out, '(S R) C -> R S C', R=repeat)
            else:
                img_feat_out = img_feat[self._sample_indices_adv(self.sample_length, (start_sec * self.image_freq), img_feat.shape[0], self.image_freq, self.train_freq)] # [60, 2048]
            au_feat_out = au_feat[self._sample_indices_adv(self.sample_length, start_sec, au_feat.shape[0], 1, self.train_freq)] # [60, 128]
            labels = self.load_lables(self.emotions, self.sample_length * (self.val_freq // self.train_freq),  vid_start_idx + (start_sec * 6), vid_end_idx, self.val_freq) # [60, 15]
            img_feat_list.append(img_feat_out)
            au_feat_list.append(au_feat_out)
            if self.cls_indices:
                labels = labels[:, self.cls_indices]
            labels_list.append(labels)
            start_sec += (self.sample_length // self.train_freq) # step the correspond time step
        assert len(img_feat_list) == len(au_feat_list) and len(au_feat_list) == len(labels_list)
        return img_feat_list, au_feat_list, labels_list, frame_count


    def get_train_item(self, index):
        # return a 60 seconds sequence inside a video
        vid, start_idx = self.vidmap_list[index]
        vid_start_idx = int(start_idx)
        vid_end_idx = int(self.vidmap_list[index + 1][1]) if index + 1 < len(self.vidmap_list) else len(self.emotions)

        img_feat = np.asarray(self.image_features[vid])
        au_feat = np.asarray(self.audio_features[vid])
        img_feat_len = img_feat.shape[0]

        # image, audio and lables shares the same start index (seconds)
        feat_start_sec = self.gen_start_idx(img_feat_len, self.image_freq)
        # print(img_feat_len, au_feat_len, feat_start_idx)
        img_feat, au_feat, labels = self.sample_item(feat_start_sec, img_feat, au_feat, self.emotions, vid_start_idx, vid_end_idx, self.train_freq)
        if self.filter == 'butter':
            b, a = signal.butter(2, 0.2, analog=False)
            flabels = signal.filtfilt(b ,a, labels, axis=0)
            flabels = np.maximum(0.0, flabels)
            # labels = np.maximum(flabels, labels) T C
        elif self.filter == 'median':
            flabels = signal.medfilt(labels, [3, 1])
            labels = flabels
        elif self.filter == 'gaussian':
            flabels = ndimage.gaussian_filter1d(labels, 0.8, axis=0)
            labels = flabels
        if self.cls_indices:
            labels = labels[:, self.cls_indices]
        return img_feat, au_feat, labels
    
    
    def sample_item(self, start_sec, img_feat, au_feat, labels, vid_start_idx, vid_end_idx, output_freq):
        # start_index: the start second in video
        # output_freq: output frequncy
        # sample {self.sample_length} frames form the given feature
        img_feat = img_feat[self._sample_indices_adv(self.sample_length, (start_sec * self.image_freq), img_feat.shape[0], self.image_freq, output_freq)] # [60, 2048]
        au_feat = au_feat[self._sample_indices_adv(self.sample_length, start_sec, au_feat.shape[0], 1, output_freq)] # [60, 128]
        # sample lables
        labels = self.load_lables(labels, self.sample_length, vid_start_idx + (start_sec * 6), vid_end_idx, output_freq) # [60, 15]
        return img_feat, au_feat, labels


    def gen_start_idx(self, frame_count, freq):
        # return the start second for the sample
        start_pos = max(0, (frame_count - self.sample_length) // freq) # start pos in seconds
        # return np.random.randint(0, start_pos + 1) # rand[)
        # numpy RNG is not stable when forked https://github.com/pytorch/pytorch/issues/5059
        # use torch random instead
        return torch.randint(0, start_pos + 1, (1,1)).item()

    
    def _sample_indices_adv(self, sample_length, start_idx, end_idx, input_freq, output_freq):
        sample_length = int(sample_length)
        if input_freq < output_freq:
            # repetition is used to fill in the gaps
            repeat = int(output_freq // input_freq)
            indices = []
            for x in range(sample_length // repeat):
                next_idx = (start_idx + x) if (start_idx + x) < end_idx else (end_idx - 1)
                next_idx = int(next_idx)
                indices.extend([next_idx] * repeat)
        else:
            step = int(input_freq // output_freq)
            indices = [int(start_idx + x * step) if (start_idx + x * step) < end_idx else int(end_idx - step) for x in range(sample_length)]
        return indices


    def load_lables(self, labels, sample_length, vid_start_idx, vid_end_idx, output_freq):
        # if vid_end_idx == vid_start_idx:
        #     return np.asarray([self.emotions[vid_start_idx]])
        assert vid_start_idx <= vid_end_idx, '{} {}'.format(vid_start_idx, vid_end_idx)
        indices = self._sample_indices_adv(sample_length, vid_start_idx, vid_end_idx, 6, output_freq)
        labels = labels[indices]        
        return labels

    



