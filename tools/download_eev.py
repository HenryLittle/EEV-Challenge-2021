import logging
import time, os, random
import pandas as pd
import csv
import argparse
from utils import parallel_process

logging.basicConfig(filename='log/download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)

CSV_ROOT_PATH = '/home/kezhou/EEV/eev-csv'
CSV_FILES = ['train.csv', 'val.csv']
OUTPUT_PATH = '/home/kezhou/EEV/data'

# 3061 vids in train [33 missing]
# 755 vids in validation [5 missing]
# 1337 vids in test [30 missing]
# *missing: [Unavailable, Private]
# total = 3816 (train + val) + 1337 (test) = 5153
# expected [Total Size]: 477GB

parser = argparse.ArgumentParser()
parser.add_argument('--input-list', type=str, default='')
parser.add_argument('--find-missing', type=int, choices=[0, 1], default=None, const=0, nargs='?', help='0: train/val, 1: test') # 0: train/val 1:test
parser.add_argument('--download-tests', action='store_true', default=False)
parser.add_argument('-j', '--num-thread', type=int, default=8)
parser.add_argument('--gen-vidmap', action='store_true', default=False)

def download_by_youtube_id(vid, output_format='mp4'):

    possible_ext = ['.mp4']
    for ext in possible_ext:
        if os.path.exists(OUTPUT_PATH + '/' + vid + ext):
            print('Skipping:', vid)
            return # skip existed file

    cmd = 'youtube-dl "https://www.youtube.com/watch?v={}" -o "{}/%(id)s.{}" --merge-output-format {}'
    cmd = cmd.format(vid, OUTPUT_PATH, output_format, output_format)
    # print(cmd)
    rv = os.system(cmd)

    if rv: # download failed
        logging.error(vid)
    # avoid spamming host
    time.sleep(random.uniform(1.0, 1.5))


def find_missing(vids, format='mp4'):
    # vids = set(['{}.{}'.format(x, vids) for x in vids])
    existed_files = os.listdir(OUTPUT_PATH)
    existed_vids = set([x[:x.find('.')] for x in existed_files])
    diff = list(set(vids) - existed_vids)
    print(len(diff), 'videos are missing.')
    return diff


def download_csv(files, column, num_thread):
    # download train validation
    for file in files:
        content = pd.read_csv(os.path.join(CSV_ROOT_PATH, file))
        print('entry count:', len(content[column].to_list()))
        print(content.shape[0])
        vids = list(set(content[column].to_list()))

        print('video count:', len(vids))

        parallel_process(vids, download_by_youtube_id, n_jobs=8)

    # check missing
    find_missing_csv(column=column, files=files)


def find_missing_csv(column='YouTube ID', files=CSV_FILES):
    missing = []
    for file in files:
        content = pd.read_csv(os.path.join(CSV_ROOT_PATH, file))
        vids = list(set(content[column].to_list()))

        print('Expected videos:', len(vids))

        missing.extend(find_missing(vids))
       
    # check missing
    with open('missing.txt', 'w') as file:
        file.write('\n'.join(missing))

def gen_vidmap_csv(column='YouTube ID', files=CSV_FILES):
    for file in files:
        filename = os.path.splitext(file)[0]
        content = pd.read_csv(os.path.join(CSV_ROOT_PATH, file))
        entries = content[column].to_list()
        idx = 0
        first = True
        vid_map = []
        prev_vid = None
        for i, vid in enumerate(entries):
            if prev_vid != vid:
                vid_map.append('%s %d' % (vid, i))
                prev_vid = vid
            
        with open('vidmap_%s.txt' % filename, 'w') as file:
            file.write('\n'.join(vid_map))

            

if __name__ == '__main__':
    args = parser.parse_args()
    if args.input_list:
        if os.path.exists(args.input_list):
            with open(args.input_list, 'r') as file:
                vids = file.readlines()
                print(len(vids))
                parallel_process(vids, download_by_youtube_id, n_jobs=args.num_thread)
    elif args.find_missing != None:
        print('Find missing videos:', ['Train/val', 'Test'][args.find_missing])
        find_missing_csv(column=['YouTube ID', 'Video ID'][args.find_missing], files=[['train.csv', 'val.csv'], ['test.csv']][args.find_missing])
    elif args.download_tests:
        print('Downloading test...')
        download_csv(files=['test.csv'], column='Video ID', num_thread=args.num_thread)
    elif args.gen_vidmap:
        print('Generate vid to index map')
        gen_vidmap_csv(files=['train.csv', 'val.csv'])
    else:
        print('Downloading train and validation...')
        download_csv(files=['train.csv', 'val.csv'], column='YouTube ID', num_thread=args.num_thread)

