#!/bin/bash
#SBATCH --job-name=mem-experiment

experiment_name="$1"
experiment_python_script=$2
chunk_size=$3
time_required_per_array_job=$4
output_dir=$5
additional_test_arguments=$6
chunk_id=$SLURM_ARRAY_TASK_ID

echo "experiment_name:" $experiment_name
echo "experiment_python_script:" $experiment_python_script
echo "chunk_size:" $chunk_size
echo "time_required_per_array_job:" $time_required_per_array_job
echo "output_dir:" $output_dir
echo "additional_test_arguments:" $additional_test_arguments
echo "chunk_id:" $chunk_id

indexes=`seq $(( chunk_id*chunk_size )) $(( (chunk_id+1)*chunk_size - 1 ))`
echo "indexes:" $indexes

for i in $indexes
do
    srun --time=$time_required_per_array_job python $experiment_python_script $experiment_name $i $output_dir $additional_test_arguments &> ${output_dir}out_$i.txt
done