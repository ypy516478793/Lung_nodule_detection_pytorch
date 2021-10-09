
cd ../
PYTHONPATH=$(pwd)

echo ""
echo "Start testing"
python detector_ben/detect.py \
    -d=luna --test=True --inference=False --gpu="4, 5" \
    --resume=detector/resmodel/res18fd9020.ckpt \
    --save-dir=luna_pretrainedLUNA

#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=luna \
#    --result_dir=detector_ben/results/luna_pretrainedLUNA/ \
#    --extra_str=minMax

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=luna \
    --result_dir=detector_ben/results/luna_pretrainedLUNA/ \
    --extra_str=fair