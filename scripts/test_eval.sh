
#cd ../detector_ben/
#python detect.py -d=methodistFull --test=True --gpu="0,3" --inference=False \
#    -s="/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210429-200058-train" \
#    -re="/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210429-200058-train/012.ckpt"

cd ../evaluate_ben/
python evaluate_result.py \
    --result_dir="/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210429-200058-train" \
    --data_dir="/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug"