lr1=0.03182
lr2=0.06018
lr3=0.03641
lr4=0.03161
lr5=0.04072

seed=159753

bs=128
epochs=150

for lr in $lr1 $lr2 $lr3 $lr4 $lr5; do
    export WANDB_NAME="exp-${RANDOM}-lr=$lr"
    python main.py data.batch_size=$bs optim.lr=$lr trainer.epochs=$epochs seed=$seed
done