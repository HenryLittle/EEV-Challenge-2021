import os
import subprocess
import argparse
import json
# import logging
import time
from multiprocessing.pool import ThreadPool
import threading
from utils import parallel_process


parser = argparse.ArgumentParser()
parser.add_argument('-j', '--num-thread', type=int, default=24)

VIDEO_ROOT = '/data0/EEV/data'
FRAME_ROOT = '/data/EEV/data-audio'


# logging.basicConfig(filename='log/ea_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)

def extract(filename):
    """ 
        filename: 12736.mp4 
    """
    assert len(filename) == 15
    video_id = filename[: -4]
    
    full_path = os.path.join(FRAME_ROOT, video_id)
    # if os.path.exists(full_path + '.wav'):
    #     return
    
    cmd = 'ffmpeg -i {}/{} -threads 1 -vn -acodec pcm_s16le -ac 1 -ar 16000 {}/{}.wav'.format(VIDEO_ROOT, filename, FRAME_ROOT, video_id)

    # extract audio
    # print(cmd.split(' ')) , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    f = subprocess.run(cmd.split(' '))
    




if __name__ == '__main__':
    args = parser.parse_args()
    filenames = os.listdir(VIDEO_ROOT)
    # extract('IHRncab3Cdg.mp4')
    parallel_process(filenames, extract, n_jobs=args.num_thread)
