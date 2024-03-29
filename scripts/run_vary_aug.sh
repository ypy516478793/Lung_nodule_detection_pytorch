#!/usr/bin/env bash

cd ../detector_ben/
GPU_ID=2
RSEED=128
FLIP=False
SWAP=False
SCALE=False
ROTATE=False
CONTRAST=False
BRIGHT=True
SHARP=False
SPLICE=False
AUGSTR=Bright

#{"flip": False, "swap": False, "scale": False, "rotate": False}

echo "Start training!"
python detect.py \
    -d=methodistFull \
    --test=False \
    --inference=False \
    --gpu=${GPU_ID} \
    --workers=32 \
    -b=8 \
    --lr=0.001 \
    --rseed=${RSEED} \
    --flip=${FLIP} \
    --swap=${SWAP} \
    --scale=${SCALE} \
    --rotate=${ROTATE} \
    --contrast=${CONTRAST} \
    --bright=${BRIGHT} \
    --sharp=${SHARP} \
    --splice=${SPLICE} \
    --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR}

echo ""
echo "Start test"
python detect.py \
    -d=methodistFull \
    --test=True \
    --inference=False \
    --gpu=${GPU_ID} \
    --workers=32 \
    -b=8 \
    --lr=0.001 \
    --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR} \
    --resume=results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR} \
    --n_test=1

echo ""
echo "Start evaluation"
cd ../evaluate_ben/
python evaluate_result.py \
    --result_dir=../detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR}/ \
    --data_dir=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/masked_with_crop/ \
    --extra_str=masked_cropped