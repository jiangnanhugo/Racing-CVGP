#!/bin/bash -l
set -x
basepath=./scibench/eureqa/
type=Feynman
nv=5
datasource=${type}_Vars${nv}
dates=2023-08-15



python parse_results.py --eureqa_path $basepath/result/$datasource/$dates/eureqa_result.csv \
--noise_type normal \
--noise_scale 0.0 \
--is_numbered 0