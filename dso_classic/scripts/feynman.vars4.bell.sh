#!/usr/bin/zsh
basepath=./scibench
py3615=python3

for pgn in FeynmanICh8Eq14.in FeynmanICh13Eq4.in FeynmanICh13Eq12.in FeynmanICh18Eq4.in FeynmanICh18Eq16.in FeynmanICh24Eq6.in FeynmanICh29Eq16.in FeynmanICh32Eq17.in FeynmanICh34Eq8.in FeynmanICh40Eq1.in FeynmanICh43Eq16.in FeynmanICh44Eq4.in FeynmanICh50Eq26.in FeynmanIICh11Eq20.in FeynmanIICh34Eq11.in FeynmanIICh35Eq18.in FeynmanIICh35Eq21.in FeynmanIICh38Eq3.in FeynmanIIICh10Eq19.in FeynmanIIICh14Eq14.in FeynmanIIICh21Eq20.in FeynmanBonus1.in FeynmanBonus3.in FeynmanBonus11.in FeynmanBonus19.in; do
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
