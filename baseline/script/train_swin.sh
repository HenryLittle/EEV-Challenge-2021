python3 main.py \
    --epochs=1000 \
    --batch-size=128 -lr=0.00020 --print-freq=100 --eval-freq=3 \
    --train-freq=1 --val-freq=6 \
    --use-multistep --step-milestones 45 80 --step-decay=0.5 \
    --train-vidmap=/data0/EEV/code/tools/vidmap_train.txt \
    --train-csv=/data0/EEV/eev-csv/train.csv \
    --val-vidmap=/data0/EEV/code/tools/vidmap_val.txt \
    --val-csv=/data0/EEV/eev-csv/val.csv \
    --img-feat-size=1536 --au-feat-size=512 \
    --image-features=/data/EEV/swin_merged.hdf5 \
    --audio-features=/data/EEV/audio-zzd-feature
    # --model=BaseImg --weight-decay=0.0 \
    # --cls-indices 1 2 \
    # --model=EmoBase \
    # --use-sam \
    # --use-sam \
    # --model=TCFPN \
    # --use-cos-wr --cos-wr-t0=40 \
    # --use-swa --swa-start=60 --cos-t-max=300 \
    # --resume=/data0/EEV/code/baseline/checkpoint/Baseline_04-17-2021_20-14/ckpt.best.pth.tar \