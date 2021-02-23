
cd ../

python detect.py \
    -d=methodistFull \
    --test=False \
    --inference=False \
    --gpu="6, 7" \
    --resume=../detector_ben/results/res18-20210106-112050_incidental/001.ckpt \
    -b=8 \
    --workers=32 \
    --save-dir=worker32_batch8_kim_masked