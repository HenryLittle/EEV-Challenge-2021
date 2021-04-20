python3 main.py \
    --run-test \
    --test-freq=6 --train-freq=1 \
    --gpu 1 --print-freq=20 --eval-freq=5\
    --resume=/data0/EEV/code/baseline/checkpoint/Baseline_04-19-2021_00-05/ckpt.best.pth.tar \
    --test-vidmap=/data0/EEV/code/tools/vidmap_test.txt \
    --image-features=/data0/EEV/feats/EEV_test_InceptionV3_6Hz.hd5f \
    --audio-features=/data0/EEV/feats/EEV_vggish_096Hz.hdf5


# Baseline_04-17-2021_20-14/ckpt.best.pth.tar 0.02617(test)