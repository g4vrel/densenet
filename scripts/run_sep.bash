wd=0.1

lr=0.01
bs=64
epochs=150
nesterov=false
momentum=false
label_smoothing=0.0

seed=159753

bs=64
epochs=100

export WANDB_NAME="sep_baseline-${RANDOM}"

python main.py data.batch_size=$bs \
    optim.lr=$lr \
    trainer.epochs=$epochs \
    seed=$seed \
    optim.weight_decay=$wd \
    optim.name=adamw \
    model.separable_convs=true
