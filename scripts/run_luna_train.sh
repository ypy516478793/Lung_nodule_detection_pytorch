
cd ../
PYTHONPATH=$(pwd)

### LUNA pretrained model test
#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=luna \
#    --result_dir=detector_ben/results/luna_pretrainedLUNA/ \
#    --extra_str=minMax


### Train using mode norm data
#echo ""
#echo "Start training"
#python detector_ben/detect.py \
#    -d=luna --test=False --inference=False --gpu="4, 5" \
#    --save-dir=luna_modeNorm
#
#echo ""
#echo "Start testing"
#python detector_ben/detect.py \
#    -d=luna --test=True --inference=False --gpu="4, 5" \
#    --resume=detector_ben/results/luna_modeNorm/ \
#    --save-dir=luna_modeNorm
#
#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=luna \
#    --result_dir=detector_ben/results/luna_modeNorm/ \
#    --extra_str=fair



## Train using preprocessed data
echo ""
echo "Start training"
python detector_ben/detect.py \
    -d=luna --test=False --inference=False --gpu="0, 1" \
    --save-dir=luna_preprocessed -b=64 --resume=064.ckpt

echo ""
echo "Start testing"
python detector_ben/detect.py \
    -d=luna --test=True --inference=False --gpu="0, 1" \
    --resume=detector_ben/results/luna_preprocessed/ \
    --save-dir=luna_preprocessed -b=32

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=luna \
    --result_dir=detector_ben/results/luna_preprocessed/ \
    --extra_str=ori
