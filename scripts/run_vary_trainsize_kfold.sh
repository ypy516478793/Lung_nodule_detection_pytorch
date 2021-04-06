#!/usr/bin/env bash

cd ../detector_ben/
GPU_ID=0
RSEED=42
KFOLD=5
LIMIT_TRAIN=1.0

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
        --limit_train=${LIMIT_TRAIN} \
        --kfold=${KFOLD} \
        --split_id=${i} \
        --save-dir=worker32_batch8_ben_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN}_${KFOLD}fold_${i}

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
        --save-dir=worker32_batch8_ben_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN}_${KFOLD}fold_${i} \
        --resume=results/worker32_batch8_ben_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN}_${KFOLD}fold_${i} \
        --n_test=1

    echo ""
    echo "Start evaluation"
    cd ../evaluate_ben/
    python evaluate_result.py \
        --result_dir=../detector_ben/results/worker32_batch8_ben_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN}_${KFOLD}fold_${i}/ \
        --data_dir=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_Ben/labeled/ \
        --extra_str=masked_cropped

    cd ../detector_ben/
done


echo ""
echo "Plot average FROC"
cd ../evaluate_ben/
python merge_froc.py \
  --root_dir=../detector_ben/results/worker32_batch8_ben_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN} \
  --save_dir=new_labels/limit1_5fold \
  --kfold=5

#echo ""
#echo "Start evaluation"
#cd ../evaluate_ben/
#python evaluate_result.py \
#    --kfold=${KFOLD}
#    --result_dir=../detector_ben/results/worker32_batch8_ben_nonPET_lr001_rs${RSEED}_limit${LIMIT_TRAIN}_${KFOLD}fold/ \
#    --data_dir=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_Ben/labeled/ \
#    --extra_str=masked_cropped