python3 main.py \
    --epochs=1000 \
    --batch-size=128 -lr=0.0002 --print-freq=100 --eval-freq=3 \
    --train-freq=1 --val-freq=6 \
    --use-multistep --step-milestones 40 80 --step-decay=0.3 \
    --train-vidmap=/data0/EEV/code/tools/vidmap_train.txt \
    --train-csv=/data0/EEV/eev-csv/train.csv \
    --val-vidmap=/data0/EEV/code/tools/vidmap_val.txt \
    --val-csv=/data0/EEV/eev-csv/val.csv \
    --image-features=/data/EEV/se_resnet_merged.hdf5 \
    --au-feat-size=512 \
    --audio-features=/data/EEV/audio-zzd-feature
    # --audio-features=/data0/EEV/feats/EEV_vggish_096Hz.hdf5
    # --image-features=/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5 \
    # --repeat-sample --workers=1 \
    # --au-feat-size=512 \
    # --audio-features=/data/EEV/audio-zzd-feature
    # --cls-mask 0 \
    # --image-features=/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5 \
    # --cls-indices 1 2 \
    # --model=EmoBase \
    # --use-sam \
    # --use-sam \
    # --model=TCFPN \
    # --use-cos-wr --cos-wr-t0=40 \
    # --use-swa --swa-start=60 --cos-t-max=300 \
    # --resume=/data0/EEV/code/baseline/checkpoint/Baseline_04-17-2021_20-14/ckpt.best.pth.tar \