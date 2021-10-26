

cd ../
export PYTHONPATH=$(pwd)


## mode1
echo ""
echo "Start testing"
python detector_ben/detect.py \
    -d=methodistFull --test=True --inference=False --gpu="0, 1" \
    --resume=detector_ben/results/luna_modeNorm/best_052.ckpt \
    --save-dir=methodist_pretrainedLUNA_mode

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=methodistFull \
    --result_dir=detector_ben/results/methodist_pretrainedLUNA_mode/ \
    --extra_str=mode


## mode2
echo ""
echo "Start testing"
python detector_ben/detect.py \
    -d=methodistFull --test=True --inference=False --gpu="0, 1" \
    --resume=detector_ben/results/luna_modeNorm2/best_006.ckpt \
    --save-dir=methodist_pretrainedLUNA_mode2

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=methodistFull \
    --result_dir=detector_ben/results/methodist_pretrainedLUNA_mode2/ \
    --extra_str=mode


## mode3
echo ""
echo "Start testing"
python detector_ben/detect.py \
    -d=methodistFull --test=True --inference=False --gpu="0, 1" \
    --resume=detector_ben/results/luna_modeNorm3/best_036.ckpt \
    --save-dir=methodist_pretrainedLUNA_mode3

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=methodistFull \
    --result_dir=detector_ben/results/methodist_pretrainedLUNA_mode3/ \
    --extra_str=mode



#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=luna \
#    --result_dir=detector_ben/results/luna_modeNorm3/ \
#    --extra_str=mode_old \
#    --nmsthresh=[0.5]