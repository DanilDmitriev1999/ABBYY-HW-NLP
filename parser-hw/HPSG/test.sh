#!/usr/bin/env bash
python src_joint/main.py test \
--dataset ptb \
--consttest-ptb-path data/test_con.txt \
--deptest-ptb-path data/test_dep.txt \
--embedding-path data/glove.gz \
--model-path-base models/joint_cwt_best_dev=93.85_devuas=95.87_devlas=94.47
