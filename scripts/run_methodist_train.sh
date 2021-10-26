
cd ../
export PYTHONPATH=$(pwd)

## mode3
echo ""
echo "Start fine-tuning"
python detector_ben/detect.py \
    -d=methodistFull --test=False --inference=False --gpu="0, 1" \
    --resume=detector_ben/results/luna_modeNorm3/best_036.ckpt \
    --save-dir=methodist_finetuned_mode3 -b=64 --workers=32 \
    -p=./Methodist_incidental/data_Ben/modeNorm3 -pv=0

echo ""
echo "Start evaluation"
python evaluate_ben/evaluate_result.py \
    -d=methodistFull \
    --result_dir=detector_ben/results/methodist_finetuned_mode3/ \
    --extra_str=ft

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