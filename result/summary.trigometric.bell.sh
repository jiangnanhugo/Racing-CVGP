#!/bin/bash -l
set -x
basepath=./scibench
type=$1
nv=$2
nt=$3
datasource=${type}_nv${nv}_nt${nt}
dates=2023-$4

python parse_randgp_results.py --fp $basepath/result/$datasource/$dates/ \
--noise_type normal \
--noise_scale 0.0 \
--is_numbered 1 \
--max_prog 10 \
--keyword $5
