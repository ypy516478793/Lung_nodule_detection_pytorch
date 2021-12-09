
cd ../


### Prepare masked data using lung segmentation from Kelvin
#python prepare_lung/nii2npz.py \
#    -d=Methodist_incidental/data_Kelvin/Nifti_data \
#    -s=Methodist_incidental/data_Kelvin/Masked_data \
#    -m=Methodist_incidental/data_Kelvin/Lung_masks/Masks_0.6

## Prepare masked data of data_Ben using lung segmentation from Kelvin
python prepare.py prep_methodist \
    -r=Methodist_incidental/data_Ben/resampled/ \
    -s=Methodist_incidental/data_Ben/masked_data_v1/ \
    -a=Methodist_incidental/data_Ben/resampled/pos_labels_norm.csv \
    --mask=True \
    --crop=True