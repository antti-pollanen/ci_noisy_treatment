#!/bin/bash
python slurm/slurm_batch_run.py education_wage education_wage 4800

# The total amount of 4800 runs comes from:
# 4 methods
# 6 noise levels in the datasets
# 200 repetitions per each combination of the above