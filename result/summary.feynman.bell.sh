#!/bin/bash -l
set -x
basepath=./scibench
datasource=Feynman
dates=2023-06-04
metric=neg_mse
python parse_results.py --fp $basepath/result/$datasource/$dates/ \
--metric $metric \
--dso_basepath $basepath/dso_classic \
--noise_type normal \
--noise_scale 0.0 \
--is_numbered 0 \
--true_program_basepath $basepath/data/unencrypted/equations_feynman/ \
