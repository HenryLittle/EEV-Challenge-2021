python3 main.py \
    --epochs=1000 \
    --val-only \
    --resume=/data0/EEV/code/baseline/checkpoint/Baseline_04-19-2021_00-05/ckpt.best.pth.tar \
    --batch-size=128 -lr=0.0002 --print-freq=100 --eval-freq=3 \
    --train-freq=1 --val-freq=6 \
    --use-multistep --step-milestones 30 60 --step-decay=0.5 \
    --train-vidmap=/data0/EEV/code/tools/vidmap_train.txt \
    --train-csv=/data0/EEV/eev-csv/train.csv \
    --val-vidmap=/data0/EEV/code/tools/vidmap_val.txt \
    --val-csv=/data0/EEV/eev-csv/val.csv \
    --img-feat-size=2048 \
    --image-features=/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5 \
    --audio-features=/data0/EEV/feats/EEV_vggish_096Hz.hdf5

# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_01-18-18/ckpt.best.pth.tar inception + vgg 0.0121
# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_02-36-07/ckpt.best.pth.tar swin + vgg 0.0122
# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_03-08-49/ckpt.best.pth.tar swin + vgg 0.0124
# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_03-33-51/ckpt.best.pth.tar swin + vgg 0.0106
# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_03-49-48/ckpt.best.pth.tar resnet + vgg 0.0102


# /data0/EEV/code/baseline/checkpoint/Baseline_04-19-2021_00-05/ckpt.best.pth.tar inception + vgg 0.0121=>0.0114