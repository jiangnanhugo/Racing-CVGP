#!/usr/bin/zsh
basepath=./scibench
py3615=python3
#

type=sincos
nv=4
nt=46
datapath=$basepath/data/unencrypted/equations_trigometric
set -x
for pgn in {0..9};
do
	eq_name=${type}_nv${nv}_nt${nt}_prog_${pgn}.in
    echo "submit $eq_name"
    
    dump_dir=$basepath/result/${type}_nv${nv}_nt${nt}/$(date +%F)

	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	for bsl in DSR PQT VPG GPMELD; do
		echo $basepath/dso_classic/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name --noise_type 'normal' --noise_scale 0.0
		sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=8 <<EOT
#!/bin/bash -l

#SBATCH --job-name="$bsl-${eq_name}"
#SBATCH --output=$log_dir/run_${bsl}_${pgn}.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=4096MB

hostname

$py3615 -m dso.run $basepath/dso_classic/config/config_regression_${type}_${bsl}.json --equation_name $datapath/$eq_name --noise_type 'normal' --noise_scale 0.0  > $dump_dir/$pgn.${bsl}.out

EOT
	done
done

#done
