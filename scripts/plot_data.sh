## plot luna_raw
python utils.plot_data.py \
    -ni 0 -d ./LUNA16/raw_files/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd \
    -a ./LUNA16/annotations.csv -z 100

## plot luna_preprocessed
python utils.plot_data.py \
    -ni 0 -d ./LUNA16/preprocessed/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260_clean.npy

## plot luna_masked
python utils.plot_data.py \
    -ni 0 -d ./LUNA16/masked_files/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260_clean.npy


## plot methodist_resampled
python utils.plot_data.py \
    -ni 0 -d ./Methodist_incidental/data_Ben/resampled/Lung_patient002/patient002-20090310.npz \
    -a ./Methodist_incidental/data_Ben/resampled/pos_labels_norm.csv

## plot methodist_masked
python utils.plot_data.py \
    -ni 0 -d ./Methodist_incidental/data_Ben/masked/Lung_patient002/patient002-20090310_clean.npz

## plot methodist_preprocessed
python utils.plot_data.py \
    -ni 0 -d ./Methodist_incidental/data_Ben/preprocessed/Lung_patient002/patient002-20090310_clean.npz

## plot methodist_modeNorm
python utils.plot_data.py \
    -ni 0 -d ./Methodist_incidental/data_Ben/modeNorm/Lung_patient002/patient002-20090310_clean.npz

#./Methodist_incidental/data_Ben/masked/Lung_patient016/patient016-20121127_clean.npz