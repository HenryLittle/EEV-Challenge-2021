python3 main.py \
    --epochs=1000 \
    --batch-size=128 -lr=0.0004 --print-freq=20 --eval-freq=3 \
    --model=TCFPN \
    --use-sam \
    --train-freq=1 --val-freq=6 \
    --train-vidmap=/data0/EEV/code/tools/vidmap_train.txt \
    --train-csv=/data0/EEV/eev-csv/train.csv \
    --val-vidmap=/data0/EEV/code/tools/vidmap_val.txt \
    --val-csv=/data0/EEV/eev-csv/val.csv \
    --image-features=/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5 \
    --audio-features=/data0/EEV/feats/EEV_vggish_096Hz.hdf5
    # --use-cos-wr --cos-wr-t0=40 \
    # --use-swa --swa-start=60 --cos-t-max=300 \
    # --resume=/data0/EEV/code/baseline/checkpoint/Baseline_04-17-2021_20-14/ckpt.best.pth.tar \