import os
import argparse
import json
from multiprocessing.pool import ThreadPool
import threading
from utils import parallel_process


VIDEO_ROOT = '/home/kezhou/EEV/data'
FRAME_ROOT = '/data1/lkz/EEV/data-frames'
# LABEL_PATH = '/home/kezhou/WLASL/code/I3D-pwc/preprocess/nslt_2000.json'

THREAD = 8
# with open(LABEL_PATH) as f:
#     data = json.load(f)



def extract(filename):
    """ 
        filename: 12736.mp4 
    """
    # global data
    video_id = filename[: -4]
    full_path = os.path.join(FRAME_ROOT, video_id)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    else:
        imgs = os.listdir(full_path)
        # frame_count = data[video_id]['action'][2] - data[video_id]['action'][1] + 1
        # if len(imgs) != frame_count:
        #     with open('./log', 'a') as f:
        #         f.write('{} expected:{} actual:{}\n'.format(video_id, frame_count, len(imgs)))
        # else: return
    
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:320 -q:v 1 -r 2 \"{}/{}/%06d.jpg\" -8'.format(VIDEO_ROOT, filename, FRAME_ROOT, video_id)
    # extract frames
    os.system(cmd)



if __name__ == '__main__':
    filenames = os.listdir(VIDEO_ROOT)
    parallel_process(filenames, extract, n_jobs=24)
    # with ThreadPool(processes=THREAD) as pool:
    #     pool.map(extract, filenames[:10])
    