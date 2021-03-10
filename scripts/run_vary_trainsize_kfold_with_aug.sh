#!/usr/bin/env bash

cd ../detector_ben/
GPU_ID=2
RSEED=42
KFOLD=5
FLIP=True
SWAP=True
SCALE=True
ROTATE=True
CONTRAST=True
BRIGHT=True
SHARP=True
SPLICE=True
AUGSTR=All
#LIMIT_TRAIN=1.0

for (( i = 0; i < $KFOLD; i++ ))
do
    echo "Start experiment with split $i!"
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
        --kfold=${KFOLD} \
        --split_id=${i} \
        --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR}_${KFOLD}fold_${i}

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
        --kfold=${KFOLD} \
        --split_id=${i} \
        --save-dir=worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR}_${KFOLD}fold_${i} \
        --resume=results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR}_${KFOLD}fold_${i} \
        --n_test=1

    echo ""
    echo "Start evaluation"
    cd ../evaluate_ben/
    python evaluate_result.py \
        --result_dir=../detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR}_${KFOLD}fold_${i}/ \
        --data_dir=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/masked_with_crop/ \
        --extra_str=masked_cropped

    cd ../detector_ben/
done



echo ""
echo "Plot average FROC"
cd ../evaluate_ben/
python merge_froc.py \
  --root_dir=../detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_aug${AUGSTR} \
  --save_dir=augAll_5fold \
  --kfold=5

#echo ""
#echo "Start evaluation"
#cd ../evaluate_ben/
#python evaluate_result.py \
#    --kfold=${KFOLD}
#    --result_dir=../detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN}_${KFOLD}fold/ \
#    --data_dir=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/masked_with_crop/ \
#    --extra_str=masked_cropped