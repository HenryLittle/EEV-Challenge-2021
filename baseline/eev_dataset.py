import torch.utils.data as data
import pandas as pd
import numpy as np
import h5py
import torch



class EEV_Dataset(data.Dataset):
    def __init__(self, csv_path, vidmap_path, image_feat_path, audio_feat_path, mode='train', image_freq=6, sample_length=60):
        assert image_freq in [2, 6] # Hz
        self.freq = image_freq # the intrinsic sample rate of image features
        self.sample_length = sample_length # 60 total frames
        if csv_path != None:
            self.csv_content = pd.read_csv(csv_path)
            # csv data
            self.emotions = np.asarray(self.csv_content.iloc[:,2:], dtype=np.float32)
            assert len(self.emotions[0]) == 15

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        
        self.vidmap_list = [x.strip().split(' ') for x in open(vidmap_path)]
        # features
        self.image_features = h5py.File(image_feat_path, 'r') # {vid: [x, 2048]} 2Hz
        self.audio_features = h5py.File(audio_feat_path, 'r') # {vid: [x, 128]} 0.96s per sample

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
    
    def get_video_info(self, index):
        # return video id and [st,ed) index in lables
        # frame_count = ed - st (6Hz)
        vid, start_idx = self.vidmap_list[index]
        vid_start_idx = int(start_idx)
        vid_end_idx = int(self.vidmap_list[index + 1][1]) if index + 1 < len(self.vidmap_list) else len(self.emotions)
        return vid, vid_start_idx, vid_end_idx

    def get_test_item(self, index, output_freq=6):
        vid, total_frames = self.vidmap_list[index] # checked

        img_feat = np.asarray(self.image_features[vid])
        au_feat = np.asarray(self.audio_features[vid])

        img_feat_list = []
        au_feat_list = []

        start_idx = 0

        frame_count = int(total_frames) # total frames (feature freqency considered)
        while start_idx < frame_count:
            img_feat = img_feat[self._sample_indices_adv(start_idx, img_feat.shape[0], self.freq, output_freq)] # [60, 2048]
            au_feat = au_feat[self._sample_indices_adv(start_idx, au_feat.shape[0], 1, output_freq)] # 
            img_feat_list.append(img_feat)
            au_feat_list.append(au_feat)
            start_idx += self.sample_length # sample next item
        assert len(img_feat_list) == len(au_feat_list)
        return img_feat_list, au_feat_list, frame_count, vid


    def get_val_item(self, index, output_freq=1):
        # return the full sequence in 60 frames segments
        # output_freq: frequncy of the sampled frames (default to 1Hz)
        vid, vid_start_idx, vid_end_idx = self.get_video_info(index)

        img_feat = np.asarray(self.image_features[vid])
        au_feat = np.asarray(self.audio_features[vid])

        img_feat_list = []
        au_feat_list = []
        labels_list = []

        start_idx = 0

        frame_count = (img_feat.shape[0] // self.freq) * output_freq # total frames (feature freqency considered)
        while start_idx < frame_count:
            img_feat, au_feat, labels = self.sample_item(start_idx, img_feat, au_feat, vid_start_idx, vid_end_idx, output_freq)
            img_feat_list.append(img_feat)
            au_feat_list.append(au_feat)
            labels_list.append(labels)
            start_idx += self.sample_length # sample next item
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
        feat_start_idx = self.gen_start_idx(img_feat_len, self.freq)
        # print(img_feat_len, au_feat_len, feat_start_idx)
        img_feat, au_feat, labels = self.sample_item(feat_start_idx, img_feat, au_feat, vid_start_idx, vid_end_idx)
        return img_feat, au_feat, labels
    
    
    def sample_item(self, start_index, img_feat, au_feat, vid_start_idx, vid_end_idx, output_freq=1):
        # start_index: the start second in video
        # output_freq: output frequncy
        # sample {self.sample_length} frames form the given feature
        img_feat = img_feat[self._sample_indices_adv(start_index, img_feat.shape[0], self.freq, output_freq)] # [60, 2048]
        au_feat = au_feat[self._sample_indices_adv(start_index, au_feat.shape[0], 1, output_freq)] # [60, 128]
        # sample lables
        labels = self.load_lables(vid_start_idx + start_index * 6, vid_end_idx, output_freq) # [60, 15]
        return img_feat, au_feat, labels


    def gen_start_idx(self, frame_count, freq):
        start_pos = max(0, frame_count // freq - self.sample_length) # start pos in seconds
        # return np.random.randint(0, start_pos + 1) # rand[)
        # numpy RNG is not stable when forked https://github.com/pytorch/pytorch/issues/5059
        # use torch random instead
        return torch.randint(0, start_pos + 1, (1,1)).item()


    def _sample_indices(self, start_idx, frame_count, freq):
        # frame_count: frames to sample from 
        # repeat last frame
        indices = [(start_idx + x * freq) if (start_idx + x * freq) < frame_count else (frame_count - freq) for x in range(self.sample_length)]
        return indices
    
    def _sample_indices_adv(self, start_idx, frame_count, input_freq, output_freq):
        if input_freq < output_freq:
            # repetition is used to fill in the gaps
            repeat = output_freq // input_freq 
            indices = []
            for x in range(self.sample_length // repeat):
                next_idx = (start_idx + x) if (start_idx + x) < frame_count else (frame_count - 1)
                indices.extend([next_idx] * repeat)
        else:
            step = input_freq // output_freq
            indices = [(start_idx + x * step) if (start_idx + x * step) < frame_count else (frame_count - step) for x in range(self.sample_length)]
        return indices


    def load_lables(self, vid_start_idx, vid_end_idx, output_freq=1):
        # if vid_end_idx == vid_start_idx:
        #     return np.asarray([self.emotions[vid_start_idx]])
        assert vid_start_idx <= vid_end_idx
        indices = self._sample_indices_adv(vid_start_idx, vid_end_idx - vid_start_idx, 6, output_freq) 
        return self.emotions[indices]

    



