# EEV-Challenge-2021

🏆 The 2nd Place Submission to the CVPR2021-Evoked Emotion from Videos challenge.

This reposiotry contains the code base for our submission at the CVPR2021-Evoked Emotion from Videos challenge. Our method achieves a correlation score 0.04430 on the final private test set of the chanllenge.

The report can be found in arVix at [Less is More: Sparse Sampling for Dense Reaction Predictions](https://arxiv.org/abs/2106.01764)

## Requirements 

- [PyTroch](https://pytorch.org/) >= 1.71
- [Einops](https://github.com/arogozhnikov/einops) >= 0.3.0

## Getting Started

### Download the dataset

Download [EEV dataset](https://github.com/google-research-datasets/eev) from google. This will give you the list of videos with the links you need to download the raw videos (3 csv files to be exact).  We also provide a simple script to download the videos  in `/tools/download_eev.py` (requies youtube-dl).

```bash
# download train and test (please specify OUPUT_PATH and CSV_ROOT_PATH in script)
python download_eev.py
# download test
python download_eev.py --download-tests
```



### Extract features

- Audio features: use VGGish to extract. We used this pytorch [implementation](https://github.com/harritaylor/torchvggish).
- Visual features: use [Swin-L](https://github.com/microsoft/Swin-Transformer) 22K pre-trianed 

Features is stored using [h5py](https://www.h5py.org), with the following structure:

```python
{vid:feature-array}
```

### Generate mapping

We provide the extracted mapping in `/tools`, there are:

- vidmap_train.txt
- vidmap_val.txt
- vidmap_test.txt

This file maps video IDs to the start index of labels in the corresponding csv files.

You can also use the `/tools/download_eev.py` to extract mapping yourself.

```bash
python3 download_eev.py --gen-vidmap
```

### Train the model

Specify the corresponding path in `baseline/script/train_baseline.sh` 

```bash
python3 main.py \
    --epochs=200 \
    --batch-size=128 -lr=0.0002 --print-freq=100 --eval-freq=3 \
    --train-freq=1 --val-freq=6 \
    --use-multistep --step-milestones 40 80 --step-decay=0.3 \
    --train-vidmap=/PATH_TO/vidmap_train.txt \
    --train-csv=/PATH_TO/train.csv \
    --val-vidmap=/PATH_TO/vidmap_val.txt \
    --val-csv=/PATH_TO/val.csv \
    --img-feat-size=1536 --au-feat-size=512 \
    --image-features=/PATH_TO/swin_merged.hdf5 \
    --audio-features=/PATH_TO/EEV_vggish_096Hz.hdf5
```

Then run the script:

```bas
bash baseline/script/train_baseline.sh
```

### Test the model

```bash
python3 main.py \
    --run-test \
    --test-freq=6 --train-freq=1 \
    --gpu 1 --print-freq=100 --eval-freq=5\
    --resume=/PATH_TO/ckpt.pth.tar \
    --test-vidmap=/PATH_TO/vidmap_test.txt \
    --img-feat-size=1536 \
    --image-features=/PATH_TO/swin_merged.hdf5 \
    --audio-features=/PATH_TO/EEV_vggish_096Hz.hdf5
```

Then run the script:

```bas
bash baseline/script/test_baseline.sh
```

#### 


