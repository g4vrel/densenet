wd3=0.001
wd4=0.1

lr=0.01
bs=64
epochs=150
nesterov=false
momentum=false
label_smoothing=0.0

seed=159753

bs=64
epochs=100

for wd in $wd3 $wd4; do
    export WANDB_NAME="adamw-${wd}-${RANDOM}"
    python main.py data.batch_size=$bs optim.lr=$lr trainer.epochs=$epochs seed=$seed optim.weight_decay=$wd optim.name=adamw
done