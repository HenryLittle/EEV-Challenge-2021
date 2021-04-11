python3 main.py \
    --epochs=500 --batch-size=128 -lr=0.0001 \
    --sl1-beta=0.1 --gpu=0\
    --train-vidmap=/data2/lkz/EEV/eev-csv/tools/vidmap_train.txt \
    --train-csv=/data2/lkz/EEV/eev-csv/train.csv \
    --val-vidmap=/data2/lkz/EEV/eev-csv/tools/vidmap_val.txt \
    --val-csv=/data2/lkz/EEV/eev-csv/val.csv \
    --image-features=/data2/lkz/EEV/inception_v3_img_features.hdf5 \
    --audio-features=/data2/lkz/EEV/vggish_audio_features.hdf5