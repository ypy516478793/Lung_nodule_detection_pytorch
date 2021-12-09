
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
#echo ""
#echo "Start fine-tuning"
#python detector_ben/detect.py \
#    -d=methodistFull --test=False --inference=False --gpu="2, 3" \
#    --resume=detector/resmodel/res18fd9020.ckpt \
#    --save-dir=methodist_finetuned_minmax -b=64 --workers=32 \
#    -p=./Methodist_incidental/data_Ben/preprocessed -pv=170
#
#echo "Start fine-tuning test"
#python detector_ben/detect.py \
#    -d=methodistFull --test=True --inference=False --gpu="2, 3" \
#    --resume=detector_ben/results/methodist_finetuned_minmax/best_004.ckpt \
#    --save-dir=methodist_finetuned_minmax \
#    -p=./Methodist_incidental/data_Ben/preprocessed -pv=170
#
#echo ""
#echo "Start evaluation"
#python evaluate_ben/evaluate_result.py \
#    -d=methodistFull \
#    --result_dir=detector_ben/results/methodist_finetuned_minmax/ \
#    --extra_str=ft




## min-max fine-tuning
echo ""
echo "Start fine-tuning"
python detector_ben/detect.py \
    -d=methodistFull --test=False --inference=False --gpu="0, 1, 2, 3" \
    --resume=detector/resmodel/res18fd9020.ckpt \
    --save-dir=methodist_finetuned_minmax_newLungSeg -b=64 --workers=32 \
    -p=./Methodist_incidental/data_Ben/masked_data_v1 -pv=170

echo "Start fine-tuning test"
python detector_ben/detect.py \
    -d=methodistFull --test=True --inference=False --gpu="0, 1, 2, 3" \
    --resume=detector_ben/results/methodist_finetuned_minmax_newLungSeg \
    --save-dir=methodist_finetuned_minmax_newLungSeg \
    -p=./Methodist_incidental/data_Ben/masked_data_v1 -pv=170

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=methodistFull \
    --result_dir=detector_ben/results/methodist_finetuned_minmax_newLungSeg/ \
    --extra_str=ft