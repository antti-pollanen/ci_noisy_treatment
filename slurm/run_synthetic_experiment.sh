#!/bin/bash
python slurm/slurm_batch_run.py synthetic synthetic 14472

# The total amount of 14472 runs comes from:
# 3 training set sizes
# 4 methods
# 3 noise levels for dataset generation
# 67 generated datasets per noise level
# 6 repetitions per each combination of the above