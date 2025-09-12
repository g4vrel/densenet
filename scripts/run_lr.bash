lr2=0.3925
lr3=0.09150
lr4=0.03478
lr5=0.1965

seed=159753

bs=128
epochs=200

for lr in $lr1 $lr2 $lr3 $lr4 $lr5; do
    export WANDB_NAME="exp-${RANDOM}-lr=$lr-bs=$bs-3"
    python main.py data.batch_size=$bs optim.lr=$lr trainer.epochs=$epochs seed=$seed
done