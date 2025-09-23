lr=0.008
seed=159753
bs=64
epochs=150
wd=0.1
nmg=32
d=2

export WANDB_NAME="sem3-final-${RANDOM}-wd=$wd-nmg=$nmg-d=$d"
python main.py \
    data.batch_size=$bs \
    optim.lr=$lr \
    trainer.epochs=$epochs \
    seed=$seed \
    model.separable_convs=true \
    model.num_min_groups=$nmg \
    model.dilation=$d \
    trainer.label_smoothing=0.0 \
    optim.name=adamw \
    optim.weight_decay=$wd