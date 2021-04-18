import time, os, random
import pandas as pd
import argparse
import subprocess
from utils import parallel_process


CSV_ROOT_PATH = '/data0/EEV/eev-csv'
CSV_FILES = ['train.csv', 'val.csv']
OUTPUT_PATH = '/data0/EEV/data'

# 3061 vids in train [38 missing]
# 755 vids in validation [10 missing]
# 3768 / 3816
# 1337 vids in test [16 missing] 1321
# *missing: [Unavailable, Private]
# total = 3816 (train + val) + 1337 (test) = 5153
# expected [Total Size]: 477GB

parser = argparse.ArgumentParser()
# ==> Runtime Config
parser.add_argument('-j', '--num-thread', type=int, default=8)
# ==> Function Selection
parser.add_argument('--input-list', type=str, default='')
parser.add_argument('--download-tests', action='store_true', default=False)
parser.add_argument('--find-missing', type=int, choices=[0, 1], default=None, const=0, nargs='?', help='0: train/val, 1: test') # 0: train/val 1:test
parser.add_argument('--gen-vidmap', action='store_true', default=False)
parser.add_argument('--gen-vidlist', action='store_true', default=False)

def download_by_youtube_id(vid, output_format='mp4', res=720):

    possible_ext = ['.mp4']
    for ext in possible_ext:
        if os.path.exists(OUTPUT_PATH + '/' + vid + ext):
            print('Skipping:', vid)
            return # skip existed file

    cmd = 'youtube-dl "https://www.youtube.com/watch?v={vid}" -f "bestvideo[height<={res}]+bestaudio/best[height<={res}]" -o "{output_path}/%(id)s.{fmt}" --merge-output-format {fmt}'
    cmd = cmd.format(vid=vid, res=res, output_path=OUTPUT_PATH, fmt=output_format)
    # print(cmd)
    rv = os.system(cmd)
    # f = subprocess.run(cmd.split(' '), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    for file in files:
        missing = []
        content = pd.read_csv(os.path.join(CSV_ROOT_PATH, file))
        vids = list(set(content[column].to_list()))

        print('Expected videos:', len(vids))

        missing.extend(find_missing(vids))

        filename = os.path.splitext(file)[0]
        # check missing
        with open('missing_%s.txt' % (filename), 'w') as file:
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

def gen_length_vidmap_csv(column='Video ID', files=['test.csv']):
    # [vid length] pair 
    for file in files:
        filename = os.path.splitext(file)[0]
        content = pd.read_csv(os.path.join(CSV_ROOT_PATH, file))
        entries = content[column].to_list()
        vid_map = []
        prev_vid = None
        count = 0
        sum = 0
        for i, vid in enumerate(entries):
            if vid == prev_vid or prev_vid == None:
                if prev_vid == None:
                    prev_vid = vid
                count += 1
            elif vid != prev_vid:
                sum += count
                vid_map.append('%s %d' % (prev_vid, count))
                count = 1
                prev_vid = vid
        vid_map.append('%s %d' % (prev_vid, count))
        sum += count
        print(sum,'/ 2890742 total entries')
        with open('vidmap_%s.txt' % filename, 'w') as file:
            file.write('\n'.join(vid_map))


def gen_vid_list(column='Video ID', files=['test.csv']):
    for file in files:
        content = pd.read_csv(os.path.join(CSV_ROOT_PATH, file))
        vids = list(set(content[column].to_list()))

        diff = set(find_missing(vids))
        vids = list(set(vids) - diff)

        filename = os.path.splitext(file)[0]
        with open('vidlist_%s.txt' % (filename), 'w') as file:
            file.write('\n'.join(vids))

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
        # gen_vidmap_csv(files=['train.csv', 'val.csv'])
        # vid start_idx length
        gen_length_vidmap_csv(column='Video ID', files=['test.csv'])
    elif args.gen_vidlist:
        print('Generate test vid list')
        gen_vid_list()
    else:
        print('Downloading train and validation...')
        download_csv(files=['train.csv', 'val.csv'], column='YouTube ID', num_thread=args.num_thread)

