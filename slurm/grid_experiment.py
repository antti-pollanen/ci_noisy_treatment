import json
import sys

import experiments.experiment_utils as experiment_utils
import experiments.education_wage_experiment_params as education_wage_experiment_params
import experiments.synthetic_experiment_params as synthetic_experiment_params
import experiments.education_wage_runs as education_wage_runs
import experiments.synthetic_runs as synthetic_runs
import utils


if __name__ == "__main__":
    print("Test started")
    experiment_name = sys.argv[1]
    task_id = int(sys.argv[2])
    output_dir = sys.argv[3]

    if experiment_name == "synthetic":
        param_module = synthetic_experiment_params
        run_module = synthetic_runs
    elif experiment_name == "education_wage":
        param_module = education_wage_experiment_params
        run_module = education_wage_runs
    else:
        raise ValueError(f"Unknown experiment_name: {experiment_name}")

    params = experiment_utils.get_run_params_by_index(
        param_module.get_constant_params(), param_module.get_param_grid(), task_id
    )

    print("params:")
    print(params)

    utils.set_random_seeds(params["base_seed"] + task_id)

    run_func = getattr(run_module, params["run_type"])
    output = run_func(params)

    with open(
        "{}out_case_{}.json".format(output_dir, task_id),
        "w",
        encoding="utf-8",
    ) as write_file:
        json.dump(output, write_file)
