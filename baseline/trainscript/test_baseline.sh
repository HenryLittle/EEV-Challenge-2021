python3 main.py \
    --run-test \
    --gpu 1 --print-freq=20 --eval-freq=5\
    --resume=/data0/EEV/code/baseline/checkpoint/Baseline_GRU_04-14-2021_20-44/ckpt.best.pth.tar \
    --test-vidmap=/data0/EEV/code/tools/vidmap_test.txt \
    --image-features=/data0/EEV/feats/EEV_test_InceptionV3_6Hz.hd5f \
    --audio-features=/data0/EEV/feats/EEV_vggish_096Hz.hdf5