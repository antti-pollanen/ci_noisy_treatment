import os

import experiments.education_wage_runs as runs
import utils

default_params = {
    "uses_dreg": False,
    "n_train": 2153,
    "n_validate": 239,
    "n_test": 598,  # real experiment 20000
    "max_epochs": 10,  # real experiment 500
    "encoder_hidden_sizes": [10, 10],  # real experiment [26, 26, 26]
    "decoder_hidden_sizes": [10, 10],  # real experiment [26, 26, 26]
    "weight_decay": 0.001,
    "n_importance_samples": 8,  # real experiment 32
    "n_likelihood_samples": 256,  # real experiment 2048
    "num_q_annealing_epochs": 2,  # real experiment 5
    "q_initial_weight": 8,
    "lr_reducer_patience": 25,
    "lr_reducer_factor": 0.1,
    "patience_in_epochs": 45,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.97,
    "max_training_time_s": 1560,
    "optimizer": "adam",
    "sample_size_for_aid_integral": 1000,  # original experiment 10000
    "uses_known_w_sd": False,
    "batch_size": 32,
    "learning_rate": 0.001,
    "relative_w_sd": 0.2,
}


def test_education_wage_ceme_run_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_known_w_sd"] = False
    params["output_dir"] = os.path.dirname(os.path.abspath(__file__))
    runs.education_wage_ceme_run(params)


def test_education_wage_ceme_run_known_w_sd_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_known_w_sd"] = True
    runs.education_wage_ceme_run(params)


def test_education_wage_ceme_run_known_w_sd_zero_me_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_known_w_sd"] = True
    params["relative_w_sd"] = 0.0
    runs.education_wage_ceme_run(params)


def test_education_wage_mlp_run_oracle_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_t_instead_of_w"] = True
    runs.education_wage_mlp_run(params)


def test_education_wage_naive_run_naive_no_errors():
    utils.set_random_seeds(0)
    params = default_params.copy()
    params["uses_t_instead_of_w"] = False
    runs.education_wage_mlp_run(params)
