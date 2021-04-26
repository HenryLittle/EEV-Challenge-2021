python3 main.py \
    --run-test \
    --test-freq=6 --train-freq=1 \
    --gpu 1 --print-freq=100 --eval-freq=5\
    --resume=/data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-40/ckpt.pth.tar \
    --test-vidmap=/data0/EEV/code/tools/vidmap_test.txt \
    --img-feat-size=1536 \
    --image-features=/data/EEV/swin_merged.hdf5 \
    --audio-features=/data0/EEV/feats/EEV_vggish_096Hz.hdf5


    # --model=BaseImg \
# Baseline_04-17-2021_20-14/ckpt.best.pth.tar 0.02617(test)
# Baseline_04-19-2021_00-05/ckpt.best.pth.tar 0.0121(val)

# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_01-18-18/ckpt.best.pth.tar inception + vgg 0.0121
# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_02-36-07/ckpt.best.pth.tar swin + vgg 0.0122
# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_03-08-49/ckpt.best.pth.tar swin + vgg 0.0124
# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_03-33-51/ckpt.best.pth.tar swin + vgg 0.0106
# /data0/EEV/code/baseline/checkpoint/Baseline_04-25_03-49-48/ckpt.best.pth.tar resnet + vgg 0.0102


# /data0/EEV/code/baseline/checkpoint/Baseline_04-19-2021_00-05/ckpt.best.pth.tar inception + vgg 0.0121=>0.0114 0.030xx(test) 


