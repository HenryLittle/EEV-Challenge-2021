import argparse

parser = argparse.ArgumentParser(description="Models on EEV dataset")
# ==== Dataset ====
parser.add_argument('--train-vidmap', type=str) # vid to index in csv map
parser.add_argument('--train-csv', type=str)    # target emotions
parser.add_argument('--val-vidmap', type=str)
parser.add_argument('--val-csv', type=str)
parser.add_argument('--test-vidmap', type=str) # test list

# ==== Extracted features ====
parser.add_argument('--image-features', type=str)
parser.add_argument('--audio-features', type=str)
parser.add_argument('--img-feat-size', type=int, default=2048)
parser.add_argument('--au-feat-size', type=int, default=128)


# ==== Model Configs ====
parser.add_argument('--model', type=str, choices=['Baseline', 'TCFPN', 'BaseAu', 'BaseImg', 'EmoBase'], default='Baseline')
parser.add_argument('--cls-indices', type=int, nargs='+', default=None)
parser.add_argument('--cls-mask', type=int, nargs='+', default=None)


# ==== Learning Configs ====
parser.add_argument('--epochs', default=120, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR')
parser.add_argument('-wd', '--weight-decay' , default=1e-5, type=float, metavar='W')
parser.add_argument('--sl1-beta', default=0.1, type=float)
parser.add_argument('--test-freq', type=int, default=6)
parser.add_argument('--train-freq', type=float, default=1)
parser.add_argument('--val-freq', type=int, default=6)
parser.add_argument('--lp-filter', type=str, default=None)
parser.add_argument('--use-sam', action='store_true', default=False)
parser.add_argument('--use-swa', action='store_true', default=False)
parser.add_argument('--swa-start', type=int, default=80)

parser.add_argument('--use-cos', action='store_true', default=False)
parser.add_argument('--cos-t-max', type=int, default=300)

parser.add_argument('--use-cos-wr', action='store_true', default=False)
parser.add_argument('--cos-wr-t0', type=int, default=5)
parser.add_argument('--cos-wr-t-mult', type=int, default=2)

parser.add_argument('--use-multistep', action='store_true', default=False)
parser.add_argument('--step-milestones', type=int, nargs='+')
parser.add_argument('--step-decay', type=float, default=0.5)



# ===== Monitor Configs ====
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ==== Runtime Configs====
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_ckpt', type=str, default='checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--run-test', action='store_true', default=False)
parser.add_argument('--run-merge', action='store_true', default=False)
parser.add_argument('--val-only', action='store_true', default=False)
# repeat sampling instead of interpolating
parser.add_argument('--repeat-sample', action='store_true', default=False)