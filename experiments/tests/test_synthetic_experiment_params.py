import experiments.experiment_utils as experiment_utils
import experiments.synthetic_experiment_params as synthetic_experiment_params


def test_num_distinct_runs():
    param_grid = synthetic_experiment_params.get_param_grid()
    num_distinct_runs = experiment_utils.get_num_distinct_runs(param_grid)
    assert num_distinct_runs == 14472  # 201*12*6 == 14472
