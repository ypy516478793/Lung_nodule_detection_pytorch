#!/bin/bash

cd ../detector_ben/
GPU_ID=2
RSEED=24

#echo "Start training!"
#python detect.py \
#    -d=methodistFull \
#    --test=False \
#    --inference=False \
#    --gpu=${GPU_ID} \
#    --workers=32 \
#    -b=8 \
#    --lr=0.001 \
#    --rseed=${RSEED} \
#    --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}

echo ""
echo "Start test"
python detect.py \
    -d=methodistFull \
    --test=True \
    --inference=False \
    --gpu=${GPU_ID} \
    --workers=0 \
    -b=8 \
    --lr=0.001 \
    --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED} \
    --resume=results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED} \
    --n_test=1


#python detect.py \
#    -d=methodistFull \
#    --test=True \
#    --inference=False \
#    --gpu=${GPU_ID} \
#    --workers=32 \
#    -b=8 \
#    --lr=0.001 \
#    --rseed=${RSEED} \
#    --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}

echo ""
echo "Start evaluation"
cd ../evaluate_ben/
python evaluate_result.py \
    --result_dir=../detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}/ \
    --data_dir=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/masked_with_crop/ \
    --extra_str=masked_cropped