
cd ../

python detect.py
    -d=methodistFull \
    --test=True \
    --inference=True \
    --gpu="7" \
    --resume=../detector_ben/results/res18-20210106-112050_incidental/001.ckpt