
cd ../
export PYTHONPATH=$(pwd)
SAVE_DIR=luna_modeNorm
GPU="0,1"

### LUNA pretrained model test
#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=luna \
#    --result_dir=detector_ben/results/luna_pretrainedLUNA/ \
#    --extra_str=minMax


## Train using mode norm data
echo ""
echo "Start training"
python detector_ben/detect.py \
    -d=luna --test=False --inference=False --gpu=${GPU} \
    --save-dir=${SAVE_DIR} -b=64 --resume=064.ckpt --save_interval=10 \
    -p="./LUNA16/modeNorm_files/" -pv=0

echo ""
echo "Start testing"
python detector_ben/detect.py \
    -d=luna --test=True --inference=False --gpu=${GPU} \
    --resume=detector_ben/results/${SAVE_DIR}/ \
    --save-dir=${SAVE_DIR} -b=32 -p="./LUNA16/modeNorm_files/" -pv=0

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=luna \
    --result_dir=detector_ben/results/${SAVE_DIR}/ \
    --extra_str=fair



### Train using preprocessed data
#echo ""
#echo "Start training"
#python detector_ben/detect.py \
#    -d=luna --test=False --inference=False --gpu=${GPU} \
#    --save-dir=luna_preprocessed -b=64
#
#echo ""
#echo "Start testing"
#python detector_ben/detect.py \
#    -d=luna --test=True --inference=False --gpu=${GPU} \
#    --resume=detector_ben/results/luna_preprocessed/ \
#    --save-dir=luna_preprocessed -b=32
#
#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=luna \
#    --result_dir=detector_ben/results/luna_preprocessed/ \
#    --extra_str=ori
