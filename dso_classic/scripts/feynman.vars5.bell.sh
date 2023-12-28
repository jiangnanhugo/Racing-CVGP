#!/usr/bin/zsh
basepath=./scibench
py3615=python3

for pgn in FeynmanICh12Eq11.in FeynmanIICh2Eq42.in FeynmanIICh6Eq15a.in FeynmanIICh11Eq3.in FeynmanIICh11Eq17.in FeynmanIICh36Eq38.in FeynmanIIICh9Eq52.in FeynmanBonus4.in FeynmanBonus12.in FeynmanBonus13.in FeynmanBonus14.in FeynmanBonus16.in; do
	trimed_name=${pgn:7:-3}
	echo "submit $trimed_name"
	equation_name=$basepath/data/unencrypted/equations_feynman/$pgn
	dump_dir=$basepath/result/Feynman/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/Feynman/$(date +%F)
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	for bsl in DSR PQT VPG GPMELD; do
		echo $bsl, $(date +'%R/%m/%d/%Y')
		sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=8 <<EOT
#!/bin/bash -l

#SBATCH --job-name="$bsl-${trimed_name}"
#SBATCH --output=$log_dir/run_${bsl}_${pgn}.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=4096MB

hostname

$py3615 -m dso.run $basepath/dso_classic/config/config_regression_${bsl}.json --equation_name $equation_name --noise_type normal --noise_scale 0.0  > $dump_dir/$pgn.${bsl}.out

EOT
	done
done

#done
