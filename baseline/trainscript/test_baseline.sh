python3 main.py \
    --run-test \
    --gpu 1 --print-freq=20 --eval-freq=5\
    --resume=/data0/EEV/code/baseline/checkpoint/Baseline_04-16-2021_00-28/ckpt.best.bak.pth \
    --test-vidmap=/data0/EEV/code/tools/vidmap_test.txt \
    --image-features=/data0/EEV/feats/EEV_test_InceptionV3_6Hz.hd5f \
    --audio-features=/data0/EEV/feats/EEV_vggish_096Hz.hdf5