#!/bin/bash -l

basepath=./scibench

thispath=$basepath/ctrl_var_gp_nan
data_path=$basepath/data/unencrypted/equations_feynman
py3615=python3

noise_type=normal
noise_scale=0.0
metric_name=neg_mse
all_equations=`ls $data_path/Feynman*.in`
for eq_name in $all_equations;
do
    echo "Submitted $eq_name"
    short_name=$(basename "$eq_name")
    trimed_name=${short_name:7:-3}
   	dump_dir=$basepath/result/Feynman/$(date +%F)
    if [ ! -d "$dump_dir" ]
    then
    	echo "create dir: $dump_dir"
    	mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/Feynman/$(date +%F)/
	if [ ! -d "$log_dir" ]
	then
    	echo "create dir: $log_dir"
    	mkdir -p $log_dir
	fi
    $py3615  $thispath/try_gp_xyx.py --equation_name $eq_name \
        		--metric_name 'neg_mse' --noise_type $noise_type --noise_scale $noise_scale \
        		 > $dump_dir/${short_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.gp.out

	$py3615  $thispath/try_gp_xyx.py --equation_name $eq_name --expand_gp \
        		--metric_name 'neg_mse' --noise_type $noise_type --noise_scale $noise_scale \
        		 > $dump_dir/${short_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.egp.out



done

