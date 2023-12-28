#!/bin/bash -l

basepath=/scibench
type=Livermore2
nv=4

thispath=$basepath/ctrl_var_gp_nan
data_path=$basepath/data/unencrypted/equations_others
py3615=python3

noise_type=normal
noise_scale=0.0
metric_name=neg_mse
for prog in {1..25};
do
    eq_name=${type}_Vars${nv}_$prog.in
    echo "submit $eq_name"
   	dump_dir=$basepath/result/${type}_Vars${nv}/$(date +%F)
    if [ ! -d "$dump_dir" ]
    then
    	echo "create dir: $dump_dir"
    	mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	if [ ! -d "$log_dir" ]
	then
    	echo "create dir: $log_dir"
    	mkdir -p $log_dir
	fi
    sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="gp_Vars${nv}_$prog"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.gp.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=2048MB

hostname

module load anaconda


python3 $thispath/try_gp_xyx.py --equation_name $data_path/$eq_name \
        		--metric_name 'neg_mse' --noise_type $noise_type --noise_scale $noise_scale \
        		 > $dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.gp.out

EOT

	sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="egp_Vars${nv}_$prog"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.egp.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=2048MB

hostname

module load anaconda


python3 $thispath/try_gp_xyx.py --equation_name $data_path/$eq_name --expand_gp \
        		--metric_name 'neg_mse' --noise_type $noise_type --noise_scale $noise_scale \
        		 > $dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.egp.out

EOT

done

