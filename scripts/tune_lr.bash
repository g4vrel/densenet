seed=159753

python tune_lr.py --cfg-path conf/config.yaml --num-trials 50 --trial-steps 2000 --low 1e-5 --high 1e-1 --seed $seed