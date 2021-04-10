python3 main.py \
    --epochs=200 --batch-size=3 -lr=0.0005 \
    --train-vidmap=/data2/lkz/EEV/eev-csv/tools/vidmap_train.txt \
    --train-csv=/data2/lkz/EEV/eev-csv/train.csv \
    --val-vidmap=/data2/lkz/EEV/eev-csv/tools/vidmap_val.txt \
    --val-csv=/data2/lkz/EEV/eev-csv/val.csv \
    --image-features=/data2/lkz/EEV/inception_v3_img_features.hdf5 \
    --audio-features=/data2/lkz/EEV/vggish_audio_features.hdf5