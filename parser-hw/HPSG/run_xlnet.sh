#!/usr/bin/env bash
python src_joint/main.py train \
 --model-path-base models/joint_xlnet \
  --epochs 100 \
 --use-xlnet \
 --const-lada 0.8 \
 --dataset ptb \
 --embedding-path data/glove.gz \
 --model-name joint_xlnet \
 --checks-per-epoch 4 \
 --num-layers 2 \
 --learning-rate 0.00005 \
 --batch-size 100 \
 --eval-batch-size 20 \
 --subbatch-max-tokens 1000 \
 --train-ptb-path data/train_con.txt \
 --dev-ptb-path data/dev_con.txt \
 --dep-train-ptb-path data/train_dep.txt \
 --dep-dev-ptb-path data/dev_dep.txt