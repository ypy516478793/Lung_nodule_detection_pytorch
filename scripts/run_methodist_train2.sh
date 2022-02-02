
cd ../
export PYTHONPATH=$(pwd)

### mode3
#echo ""
##echo "Start fine-tuning"
##python detector_ben/detect.py \
##    -d=methodistFull --test=False --inference=False --gpu="2, 3" \
##    --resume=detector/resmodel/res18fd9020.ckpt \
##    --save-dir=methodist_finetuned_minmax -b=64 --workers=32 \
##    -p=./Methodist_incidental/data_Ben/preprocessed -pv=170
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

#python detect.py \
#    -d=methodistFull \
#    --test=False \
#    --inference=False \
#    --gpu="6, 7" \
#    --resume=../detector_ben/results/res18-20210106-112050_incidental/001.ckpt \
#    -b=8 \
#    --workers=32 \
#    --save-dir=worker32_batch8_kim_masked


#python detect.py \
#    -d=methodistFull \
#    --test=False \
#    --inference=False \
#    --gpu="6, 7" \
#    --resume=../detector_ben/results/res18-20210106-112050_incidental/001.ckpt \
#    -b=8 \
#    --workers=32 \
#    --save-dir=worker32_batch8_kim_masked



## min-max fine-tuning
# methodist_finetuned_minmax: random trainVal/test split
# methodist_finetuned_minmax_fixTest: fixed trainVal/test split
echo ""
echo "Start fine-tuning"
python detector_ben/detect.py \
    -d=methBenMinmax --test=False --inference=False --gpu="6, 7" \
    --resume=detector/resmodel/res18fd9020.ckpt \
    --save-dir=methodist_finetuned_minmax_fixTest -b=64 --workers=32 \

echo "Start fine-tuning test"
python detector_ben/detect.py \
    -d=methBenMinmax --test=True --inference=False --gpu="6, 7" \
    --resume=detector_ben/results/methodist_finetuned_minmax_fixTest \
    --save-dir=methodist_finetuned_minmax_fixTest \

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=methBenMinmax \
    --result_dir=detector_ben/results/methodist_finetuned_minmax_fixTest/ \
    --extra_str=ft
