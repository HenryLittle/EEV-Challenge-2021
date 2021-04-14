import os
import argparse
import json
from multiprocessing.pool import ThreadPool
import threading
from utils import parallel_process
import subprocess
from tqdm import tqdm

VIDEO_ROOT = '/data0/EEV/data'
FRAME_ROOT = '/data0/EEV/data-frames'
# LABEL_PATH = '/home/kezhou/WLASL/code/I3D-pwc/preprocess/nslt_2000.json'

THREAD = 8
# with open(LABEL_PATH) as f:
#     data = json.load(f)
parser = argparse.ArgumentParser()
# ==> Runtime Config
parser.add_argument('-j', '--num-thread', type=int, default=8)
# ==> Function Selection
parser.add_argument('--input-list', type=str, default='')


def extract(filename, freq=6):
    """ 
        filename: 12736.mp4 or just vid
    """
    # global data
    if len(filename) != 15:
        print('File:%s' % filename)
    video_id = os.path.splitext(filename)[0]
    full_path = os.path.join(FRAME_ROOT, video_id)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    else:
        imgs = os.listdir(full_path)
        assert len(imgs) != 0
        return 

    cmd = 'ffmpeg -i {}/{} -threads 1 -vf scale=-1:320 -q:v 1 -r {} {}/{}/%06d.jpg'.format(VIDEO_ROOT, filename, freq, FRAME_ROOT, video_id)
    # extract frames
    # os.system(cmd)
    subprocess.run(cmd.split(' '), stdout=subprocess.DEVNULL)



if __name__ == '__main__':
    args = parser.parse_args()
    if args.input_list:
        if os.path.isfile(args.input_list):
            with open(args.input_list) as file:
                filenames = file.readlines()
                filenames = [x.rstrip() for x in filenames]
                print('Input list has', len(filenames), 'videos')
        else:
            exit(1)
    else:
        filenames = os.listdir(VIDEO_ROOT)
    parallel_process(filenames, extract, n_jobs=args.num_thread)
    # for file in tqdm(filenames):
    #     extract(file)
    # with ThreadPool(processes=THREAD) as pool:
    #     pool.map(extract, filenames[:10])
    