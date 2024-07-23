import experiments.synthetic_runs as runs
import utils

default_params = {
    "gp_data_generator_params": {
        "n_total": 10000,
        "z_mean": 0,
        "z_sd": 1,
        "kernel_alpha": 1,
        "kernel_scale": 2,
        "relative_w_sd": 0.2,  # varies in real experiment
        "relative_y_sd": 0.2,  # varies in real experiment
        "local_seed": 0,
        "n_grid_points": 100,  # real experiment 1000
    },
    "uses_dreg": False,
    "n_train": 4000,  # varies in real experiment
    "n_validate": 1000,  # real experiment 8000
    "n_test": 4000,  # real experiment 20000
    "max_epochs": 10,  # real experiment 500
    "encoder_hidden_sizes": [10, 10],  # real experiment [20, 20, 20]
    "decoder_hidden_sizes": [10, 10],  # real experiment [20, 20, 20]
    "weight_decay": 0,
    "n_importance_samples": 8,  # real experiment 32
    "n_likelihood_samples": 256,  # real experiment 2048
    "num_q_annealing_epochs": 2,  # real experiment 10
    "q_initial_weight": 4,
    "lr_reducer_patience": 30,
    "lr_reducer_factor": 0.1,
    "patience_in_epochs": 40,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.97,
    "max_training_time_s": 1560,
    "optimizer": "adam",
    "sample_size_for_aid_integral": 1000,  # original experiment 10000
    "uses_known_w_sd": False,
    "batch_size": 64,  # varies in real experiment
    "learning_rate": 0.003,  # varies in real experiment
    "case_id": 0,
}


def test_synthetic_ceme_run_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_known_w_sd"] = False
    runs.synthetic_ceme_run(params)


def test_synthetic_ceme_run_known_w_sd_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_known_w_sd"] = True
    runs.synthetic_ceme_run(params)


def test_synthetic_mlp_run_oracle_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_t_instead_of_w"] = True
    runs.synthetic_mlp_run(params)


def test_synthetic_naive_run_naive_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_t_instead_of_w"] = False
    runs.synthetic_mlp_run(params)
