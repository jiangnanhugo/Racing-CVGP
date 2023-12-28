#!/bin/bash -l
set -x
basepath=./scibench
datasource=Livermore2_Vars$1
dates=2023-$2
python parse_randgp_results.py --fp $basepath/result/$datasource/$dates/ \
--noise_type normal \
--noise_scale 0.0 \
--is_numbered 1 \
--keyword treegp \
--max_prog 26



