#!/bin/bash -l

basepath=/depot/yexiang/apps/jiang631/data/scibench
type=$1
nv=$2
nt=$3

thispath=$basepath/multiorder_cvgp
data_path=$basepath/data/unencrypted/equations_trigometric
py310=/home/jiang631/workspace/miniconda3/envs/py310/bin/python

noise_type=normal
noise_scale=0.0
metric_name=neg_mse
for prog in {0..9};
do
    eq_name=${type}_nv${nv}_nt${nt}_prog_${prog}.in
    echo "submit $eq_name"
   	dump_dir=$basepath/result/${type}_nv${nv}_nt${nt}/$(date +%F)
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
	for opt in BFGS Nelder-Mead CG basinhopping dual_annealing shgo;
	do
		echo $opt
		sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="tgp_${opt}_${type}${nv}${nt}_${prog}"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.opt${opt}.treegp.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=4096MB

hostname

$py310 $thispath/main.py --equation_name $data_path/$eq_name   --optimizer $opt \
        		--metric_name 'neg_mse' --noise_type $noise_type --noise_scale $noise_scale \
        		 > $dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt${opt}.treegp.out

EOT
done
done

