import argparse

parser = argparse.ArgumentParser(description="Models on EEV dataset")
# ==== Dataset ====
parser.add_argument('--train-vidmap', type=str) # vid to index in csv map
parser.add_argument('--train-csv', type=str)    # target emotions
parser.add_argument('--val-vidmap', type=str)
parser.add_argument('--val-csv', type=str)
# ==== Extracted features ====
parser.add_argument('--image-features', type=str)
parser.add_argument('--audio-features', type=str)

# ==== Learning Configs ====
parser.add_argument('--epochs', default=120, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR')
parser.add_argument('-wd', '--weight-decay' , default=5e-4, type=float, metavar='W')
parser.add_argument('--sl1-beta', default=0.1, type=float)

# ===== Monitor Configs ====
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')

# ==== Runtime Configs====
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('-j', '--workers', default=8, type=int)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_ckpt', type=str, default='checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')