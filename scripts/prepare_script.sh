#!/usr/bin/env bash

DATA_DIR=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_unlabeled/normalized
#DATA_DIR=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/labeled

cd ../

python prepare.py get_info \
    -r=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_unlabeled/normalized

echo "<<<<< Finish get_info <<<<<"

python prepare.py ch_infopath \
    -s=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_unlabeled/normalized

echo "<<<<< Finish ch_infopath <<<<<"

python prepare.py ass_pet \
    -s=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_unlabeled/normalized

echo "<<<<< Finish ass_pet <<<<<"

python prepare.py prep_methodist \
    -s=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_unlabeled/masked_with_crop \
    -r=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_unlabeled/normalized \
    -m=True \
    -c=True

echo "<<<<< Finish prep_methodist <<<<<"

python prepare.py extract \
    -r="/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/additional_0412/raw" \
    -s="/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/additional_0412" \
    -p="/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/additional_0412/checklist.xlsx" \
    -n=False

#python prepare.py extract -s=$(dirname ${DATA_DIR}) -r=${DATA_DIR}

#python prepare.py extract \
#    -s=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/ \
#    -r=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/labeled/ \
#    -p=/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/checklist.xlsx \
#    -n=False