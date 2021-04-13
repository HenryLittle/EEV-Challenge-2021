python3 main.py \
    --epochs=1000 --batch-size=128 -lr=0.0001 --gpu 1 --print-freq=20 --eval-freq=5\
    --train-vidmap=/data0/EEV/code/tools/vidmap_train.txt \
    --train-csv=/data0/EEV/eev-csv/train.csv \
    --val-vidmap=/data0/EEV/code/tools/vidmap_val.txt \
    --val-csv=/data0/EEV/eev-csv/val.csv \
    --image-features=/data0/EEV/feats/EEV_InceptionV3_6Hz.hdf5 \
    --audio-features=/data0/EEV/vggish_audio_features.hdf5
