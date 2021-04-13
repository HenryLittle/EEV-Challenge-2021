from utils import parallel_process
from PIL import Image
import os 
import time 
from torchvision import transforms

ROOT = '/data0/EEV/data-frames/00DCWMfJIpc'

ttf = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize([ 0.485, 0.456, 0.406 ], [0.229, 0.224, 0.225])
])

def a(i):
    return Image.open(os.path.join(ROOT, i)), i[: -4]

st = time.time()

files = os.listdir('/data0/EEV/data-frames/00DCWMfJIpc')
# print(files)
res = parallel_process(files, a, n_jobs=20)
res = sorted(res, key=lambda t: t[1])

load_time = time.time() - st
st = time.time()
# print([x[1] for x in res])

res = [x[1] for x in res]

print(res)
print('1288', 'transform', time.time() - st, 'load', load_time)
# 4.6070 1377 imgs  7.2s total