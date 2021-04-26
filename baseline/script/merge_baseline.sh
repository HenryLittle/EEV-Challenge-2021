python3 main.py \
    --run-merge \
    --epochs=80 --batch-size=128 -lr=0.0002 --print-freq=20 --eval-freq=5 \
    --train-freq=1 --val-freq=6 \
    --use-multistep --step-milestones 40 80 --step-decay=0.3 \
    --train-vidmap=/data0/EEV/code/tools/vidmap_train.txt \
    --train-csv=/data0/EEV/eev-csv/train.csv \
    --val-vidmap=/data0/EEV/code/tools/vidmap_val.txt \
    --val-csv=/data0/EEV/eev-csv/val.csv \
    --img-feat-size=1536 \
    --image-features=/data/EEV/swin_merged.hdf5 \
    --audio-features=/data0/EEV/feats/EEV_vggish_096Hz.hdf5


# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-02/ckpt.pth.tar 
# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-06/ckpt.pth.tar 
# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-15/ckpt.pth.tar 
# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-20/ckpt.pth.tar 
# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-23/ckpt.pth.tar

# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-24/ckpt.pth.tar 
# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-28/ckpt.pth.tar 
# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-35/ckpt.pth.tar 
# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-39/ckpt.pth.tar 
# /data0/EEV/code/baseline/checkpoint/Baseline_merged_04-25_08-40/ckpt.pth.tar 
