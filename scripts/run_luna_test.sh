
cd ../

python detect.py \
    -d=lunaRaw \
    --test=True \
    --inference=False \
    --gpu="6, 7" \
    --resume=../detector_ben/results/res18-20210105-171908_LUNA/050.ckpt