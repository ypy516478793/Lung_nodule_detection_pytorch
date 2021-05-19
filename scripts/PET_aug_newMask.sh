#!/usr/bin/env bash

cd ../detector_ben/
DATA_DIR=/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug/
SAVE_DIR=new_labels/PET_newMask_aug_5fold
GPU_ID=2,3
RSEED=42
KFOLD=5
FLIP=True
SWAP=True
SCALE=True
ROTATE=True
AUGSTR=Regular
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
        --flip=${FLIP} \
        --swap=${SWAP} \
        --scale=${SCALE} \
        --rotate=${ROTATE} \
        --kfold=${KFOLD} \
        --split_id=${i} \
        --save-dir=${SAVE_DIR}/rs${RSEED}_aug${AUGSTR}_${KFOLD}fold_${i}

    echo ""
    echo "Start test"
    python detect.py \
        -d=methodistFull \
        --test=True \
        --inference=False \
        --gpu=${GPU_ID} \
        --workers=32 \
        -b=2 \
        --lr=0.001 \
        --kfold=${KFOLD} \
        --split_id=${i} \
        --save-dir=${SAVE_DIR}/rs${RSEED}_aug${AUGSTR}_${KFOLD}fold_${i} \
        --resume=${SAVE_DIR}/rs${RSEED}_aug${AUGSTR}_${KFOLD}fold_${i} \
        --n_test=1

    echo ""
    echo "Start evaluation"
    cd ../evaluate_ben/
    python evaluate_result.py \
        --result_dir=../detector_ben/results/${SAVE_DIR}/rs${RSEED}_aug${AUGSTR}_${KFOLD}fold_${i}/ \
        --data_dir=${DATA_DIR} \
        --extra_str=

    cd ../detector_ben/
done



echo ""
echo "Plot average FROC"
cd ../evaluate_ben/
python merge_froc.py \
  --root_dir=../detector_ben/results/${SAVE_DIR}/rs${RSEED}_aug${AUGSTR} \
  --save_dir=${SAVE_DIR} \
  --kfold=5