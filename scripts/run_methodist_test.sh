

cd ../
PYTHONPATH=$(pwd)

echo ""
echo "Start testing"
python detector_ben/detect.py \
    -d=methodistFull --test=True --inference=False --gpu="6, 7" \
    --resume=detector/resmodel/res18fd9020.ckpt \
    --save-dir=methodist_pretrainedLUNA

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=methodistFull \
    --result_dir=detector_ben/results/methodist_pretrainedLUNA/ \
    --extra_str=minMax
