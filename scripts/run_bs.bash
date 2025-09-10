export WANDB_NAME=exp-${RANDOM}-bs8
python main.py data.batch_size=8

export WANDB_NAME=exp-${RANDOM}-bs16
python main.py data.batch_size=16

export WANDB_NAME=exp-${RANDOM}-bs64
python main.py data.batch_size=64

export WANDB_NAME=exp-${RANDOM}-bs128
python main.py data.batch_size=128