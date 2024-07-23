#!/bin/bash

# Script for running automatic testing on the cluster

num_cpus=10
mem_per_cpu=4G
time="00:05:00"
test_file="" 
test_function=""

if [ -n "$test_function" ]; then
  test_opt="-k $test_function"
else
  test_opt=""
fi

srun --time=$time --mem-per-cpu=$mem_per_cpu --cpus-per-task=$num_cpus python -m pytest --forked --durations=0 -n $num_cpus -rP -v $test_file $test_opt

 