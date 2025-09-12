lr2=0.093
seed=159753
bs=64
epochs=330
dropout=0.30

export WANDB_NAME="exp-${RANDOM}-schd4"
python main.py \
  data.batch_size=$bs \
  optim.lr=$lr2 \
  trainer.epochs=$epochs \
  seed=$seed \
  model.dropout=$dropout