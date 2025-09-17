lr=0.01
seed=159753
bs=64
epochs=330
dropout=0.0
nesterov=false
momentum=false
label_smoothing=0.0
weight_decay=0.0

export WANDB_NAME="adamw-${RANDOM}"
export WANDB_NOTES="Baseline experiment with AdamW lr=$lr, seed=$seed, no regularization. Since weight decay is to be tuned, we need an procedure that makes its value independet of the lr."

python main.py \
  data.batch_size=$bs \
  optim.lr=$lr \
  trainer.epochs=$epochs \
  seed=$seed \
  model.dropout=$dropout \
  optim.weight_decay=$weight_decay \
  optim.momentum=$momentum \
  optim.nesterov=$nesterov \
  trainer.label_smoothing=$label_smoothing \
  optim.name=adamw