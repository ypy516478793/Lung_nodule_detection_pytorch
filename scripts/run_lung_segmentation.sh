#!/usr/bin/env bash

cd ../

python prepare.py prep_methodist \
    -s="/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug" \
    -r="/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/labeled" \
    -m=True \
    -c=True

cd data/
python inspect_data.py