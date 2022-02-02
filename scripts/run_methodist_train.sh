
cd ../
export PYTHONPATH=$(pwd)

### mode3 fine-turning
#echo ""
#echo "Start fine-tuning"
#python detector_ben/detect.py \
#    -d=methodistFull --test=False --inference=False --gpu="0, 1" \
#    --resume=detector_ben/results/luna_modeNorm3/best_036.ckpt \
#    --save-dir=methodist_finetuned_mode3 -b=64 --workers=32 \
#    -p=./Methodist_incidental/data_Ben/modeNorm3 -pv=0
#
#echo "Start fine-tuning test"
#python detector_ben/detect.py \
#    -d=methodistFull --test=True --inference=False --gpu="0, 1" \
#    --resume=detector_ben/results/methodist_finetuned_mode3/best_003.ckpt \
#    --save-dir=methodist_finetuned_mode3 \
#    -p=./Methodist_incidental/data_Ben/modeNorm3 -pv=0
#
#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=methodistFull \
#    --result_dir=detector_ben/results/methodist_finetuned_mode3/ \
#    --extra_str=ft


### min-max fine-tuning
## methodist_finetuned_minmax: random trainVal/test split
## methodist_finetuned_minmax_fixTest: fixed trainVal/test split
#echo ""
#echo "Start fine-tuning"
#python detector_ben/detect.py \
#    -d=methBenMinmax --test=False --inference=False --gpu="4, 5, 6, 7" \
#    --resume=detector/resmodel/res18fd9020.ckpt \
#    --save-dir=methodist_finetuned_minmax_fixTest -b=64 --workers=32 \
#
#echo "Start fine-tuning test"
#python detector_ben/detect.py \
#    -d=methBenMinmax --test=True --inference=False --gpu="4, 5, 6, 7" \
#    --resume=detector_ben/results/methodist_finetuned_minmax_fixTest \
#    --save-dir=methodist_finetuned_minmax_fixTest \
#
#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=methBenMinmax \
#    --result_dir=detector_ben/results/methodist_finetuned_minmax_fixTest/ \
#    --extra_str=ft




## min-max fine-tuning newLungSeg
# methodist_finetuned_minmax_newLungSeg: random trainVal/test split
# methodist_finetuned_minmax_newLungSeg_fixTest: fixed trainVal/test split
echo ""
echo "Start fine-tuning"
python detector_ben/detect.py \
    -d=methBenMinmaxNew --test=False --inference=False --gpu="6, 7" \
    --resume=detector/resmodel/res18fd9020.ckpt \
    --save-dir=methodist_finetuned_minmax_newLungSeg_fixTest -b=64 --workers=32

echo "Start fine-tuning test"
python detector_ben/detect.py \
    -d=methBenMinmaxNew --test=True --inference=False --gpu="6, 7" \
    --resume=detector_ben/results/methodist_finetuned_minmax_newLungSeg_fixTest \
    --save-dir=methodist_finetuned_minmax_newLungSeg_fixTest \

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=methBenMinmaxNew \
    --result_dir=detector_ben/results/methodist_finetuned_minmax_newLungSeg_fixTest/ \
    --extra_str=ft