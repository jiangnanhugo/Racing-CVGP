#!/bin/bash -l
set -x
basepath=./scibench/eureqa/
type=sincos
nv=8
nt=812
datasource=${type}_nv${nv}_nt${nt}
dates=2023-08-06
metric=neg_mse


python parse_results.py --eureqa_path $basepath/result/$datasource/$dates/eureqa_result.csv \
--metric $metric \
--noise_type normal \
--noise_scale 0.0
