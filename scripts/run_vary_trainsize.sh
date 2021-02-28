#!/usr/bin/env bash

cd ../detector_ben/
GPU_ID=1
RSEED=128
LIMIT_TRAIN=1.0

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
    --limit_train=${LIMIT_TRAIN} \
    --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN}

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
    --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN} \
    --resume=results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN} \
    --n_test=1

echo ""
echo "Start evaluation"
cd ../evaluate_ben/
python evaluate_result.py \
    --result_dir=../detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN}/ \
    --data_dir=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/masked_with_crop/ \
    --extra_str=masked_cropped