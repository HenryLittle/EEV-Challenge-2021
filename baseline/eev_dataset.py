import torch.utils.data as data
import pandas as pd
import numpy as np
import h5py
import torch

DIFF = ['-F8IID_1rfc', '0JUyvE1Ns5M', '0c-2bxQsoi0', '0lLptaVPNko', '0qOBK1uD-1A', '0uZjkmX_CDE', '0vKJ-4SS27Q', '0yhtOii0tAc', '1BObSxLpj10', '1C4gCQj4J10', '1HOhvrfrAoQ', '1_2UzFD1mUo', '1lRiKyDWC5o', '29l5I3umkmA', '3Dnvm4dUQFo', '3_v_RusQzO8', '3b55dySuCi8', '427Nv5ApfOo', '4R27wVIz5mI', '4Vd1Z2_1ix0', '4hEkmOv92Pw', '4ojCt-GJ40I', '5AxvXjsEufQ', '5Qc24tEQcOM', '5SnV3gfv1tY', '5uzsmfDsmQk', '5vlaxMUX65Q', '6BXuC5wXea0', '6I6wWzXvKyQ', '6lbc7HLA0qw', '87jeCmqWCVo', '8RueXDLwUDs', '8UYvrDhloII', '8XXb67uo19Y', '8wit4qkiQFk', '9-02PtesPPQ', '9QXdZKd3IEw', '9TjfkXmwbTs', '9YXOgFNT844', '9vqeaw_UUcQ', 'A37EwJNIuxw', 'AWmqvvdBbJk', 'AaPsD2GCBvM', 'AbIwhNX762s', 'BQ_0wrgXT2k', 'CGsaHgQ6B_M', 'CO_xfJHu7SI', 'CT3lwwrQfAA', 'Cj5Uol8_x1k', 'Ckk6w-6kYOc', 'D118FwWhOSM', 'DcEP5pkY18g', 'DrbJizIbqpE', 'EBNunyb6QRk', 'EDSncomOn7M', 'ENlk5NdHVxM', 'EcVqFlz8XXQ', 'EfjYTYWE1FQ', 'EtD6Z4UXl9o', 'FUUZ9s7KVpY', 'FXR94rFoFs0', 'FZmcllm17fo', 'Fa5kcMPD3Ks', 'Ffln-0nfRFM', 'FrGRFdDzkFM', 'GXWXKnleUyk', 'H91i8v0F9IA', 'Ha1mGHzwoZI', 'I-dVdjn1XDM', 'I8kS8WhwMWY', 'I8q2eutBoXk', 'Id13Or3e8KE', 'IyWr5OJxSzQ', 'J-JWUX-Zs1c', 'JGdCnc1344Q', 'JIXh5tkH-44', 'JMC22AVjkrU', 'JU7fjMCk4IA', 'JXO3zM9Jdks', 'KCscz09IJ9s', 'L08LjkN1k70', 'LLYB5lGR6H0', 'Li4vN5v5MQA', 'Md6iLKwJmLM', 'MjnCsXgEiBs', 'N1g3pL-i9eA', 'N5-yVpplDG8', 'NTHgj1HvC94', 'NXF3SbnWvNM', 'OC7oe6ciDzI', 'OH3hGUYT7DU', 'Oi_tKiEBBZo', 'OnJNxOYK7zE', 'P4iIxhxoOnw', 'P7lhffZT63I', 'PYC0tv5Q70Y', 'Q1p6wOWFuyU', 'QK07sDtOHUg', 'QpGQel9AXzU', 'Qq1wHFXdnaI', 'RaaEtS-Wi0I', 'RcfGvPLhpEA', 'RnO-qDhAAZ0', 'SQEmS3PdATY', 'ScCaKfXUzvA', 'StSx1zd-qcw', 'T-lU0PFspuY', 'T5yNZ80FHjo', 'TKRrnv5r-x4', 'TnjYnE93NfU', 'TyrCF89pwXs', 'UGliQxL6cco', 'VGodAJkueFc', 'VNmzNQWgIuc', 'VSBCtAyBzSc', 'Vbb-RRtWtn0', 'VqXckVbbIlQ', 'WhNKs72ky1c', 'XP5dxF12VpQ', 'Xu_zmFP1q-0', 'YQBugJ9lB6A', 'YQJz1aNBkv0', 'YWWbFydeOuU', 'Z-VxGSCfRQ0', 'Z9tvc6GvZUA', 'ZKArWe85ZXo', 'ZW7t5cayPWA', 'a3X_0RS5a8k', 'aJG7yPu44v0', 'b8g1WOz8HWo', 'bLY2NgK7yvw', 'bYqC5rojt6I', 'cC75TF4Be1M', 'cPO7F2WXABw', 'cVW9GTUKDZM', 'cVjl2dhMXc8', 'dHWJxhKQNsA', 'ebybqir7AM0', 'eclVcwon99E', 'eu9UtVaWYaI', 'f1CaK1VH4RM', 'fRuFLppTxug', 'g3On93DpNyo', 'gUHL90du3sA', 'gV7u6NYyLpA', 'gWs-CwvcoRo', 'hEZqxlUQ5Nk', 'i-vyBhEK36o', 'iroQf2P8TP8', 'j1htCrlZ46k', 'jJjs3Xd3JY0', 'jVewZoL5T2c', 'jcgidDSrehs', 'jnnzbdt4_RE', 'jrPTHnpNZgk', 'kmFM_4oUAlo', 'kzzdg4yYkqg', 'l2eKVkJNv1s', 'l2gmLef-N0Y', 'l7RNRfs2EJI', 'l7Ww0rdKjlE', 'lLUPvgQNqBE', 'lMjyqZVYVp4', 'liPqMGBGO40', 'mCKtiJHX1Bs', 'mQF3WAFof6g', 'muOibQWLz3o', 'n0c983DYcl0', 'nPnz-u6grw4', 'nygs0imMQqU', 'o6FOCHl7pU4', 'oZS3fekWNOM', 'olDhE5LPf84', 'ovVEVD7knzA', 'oveDwY9VEJc', 'p-msiddySUI', 'pNXal3Z0qd4', 'pY1PMX63x9s', 'q0rfMYM5YDQ', 'q4yQ4jBZMHw', 'q5iNr6usKoA', 'qMQvzMeqn9E', 'qdjqCGJQVow', 'qreaOSjzDh8', 'r17o5eS-csg', 'rpuCUnKowMU', 'sBH3_RXQi5w', 'sOvJ4vpBND8', 'sSHPoM0rTbI', 't9nQCsFIACg', 'tUcz3znVpuc', 'tb-VOIWDTp0', 'tdmnuAL4Nas', 'tjFqmidggN8', 'uBd6eK06PjM', 'uFdAaIc1308', 'uIASUtVS170', 'uKVt0v6uRI0', 'uLMpnb-HWLE', 'uVnSq8X3G90', 'vOacARxLYZQ', 'vdYDweQdML4', 'vepCkXLrnxs', 'wrBNSi1rIbc', 'x77fRikiMiw', 'xRwWM0IJwz0', 'y0WnvBCFbMo', 'yHZYI0WjgmM', 'yKi45mhvOc4', 'ygn_x99uKOE', 'zTGDsm9Q77U', 'zYpnOPF2O2o']

class EEV_Dataset(data.Dataset):
    def __init__(self, csv_path, vidmap_path, image_feat_path, audio_feat_path, mode='train', image_freq=6, sample_length=60):
        global DIFF
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
        print('vidmap list filter length:', len(self.vidmap_list))

 

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
        vid, _ = self.vidmap_list[index]

        img_feat = np.asarray(self.image_features[vid])
        au_feat = np.asarray(self.audio_features[vid])

        img_feat_list = []
        au_feat_list = []

        start_idx = 0

        frame_count = (img_feat.shape[0] // self.freq) * output_freq # total frames (feature freqency considered)
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

    



