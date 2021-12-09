
cd ../nodcls/
PYTHONPATH=$(pwd)


## Train using preprocessed data
echo ""
echo "Start training"
python main_nodcls.py

#echo ""
#echo "Start testing"
#python detector_ben/detect.py \
#    -d=luna --test=True --inference=False --gpu="0, 1" \
#    --resume=detector_ben/results/luna_preprocessed/ \
#    --save-dir=luna_preprocessed -b=32
#
#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=luna \
#    --result_dir=detector_ben/results/luna_preprocessed/ \
#    --extra_str=ori
