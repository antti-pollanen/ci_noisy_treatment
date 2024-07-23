from typing import Dict, List, Any


def get_constant_params() -> Dict[str, Any]:
    constant_params: Dict[str, Any] = {
        "batch_size": 32,
        "n_train": 2153,
        "n_validate": 239,
        "n_test": 598,
        "max_epochs": 500,
        "encoder_hidden_sizes": [26, 26, 26],
        "decoder_hidden_sizes": [26, 26, 26],
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "n_importance_samples": 32,
        "n_likelihood_samples": 2048,
        "num_q_annealing_epochs": 5,
        "q_initial_weight": 8,
        "lr_reducer_patience": 25,
        "lr_reducer_factor": 0.1,
        "patience_in_epochs": 45,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.97,
        "max_training_time_s": 1560,
        "base_seed": 0,
    }
    return constant_params


def get_param_grid() -> Dict[str, List[Dict[str, Any]]]:
    param_grid: Dict[str, List[Dict[str, Any]]] = {
        "algorithm": [
            {
                "name": "CEME",
                "uses_known_w_sd": False,
                "run_type": "education_wage_ceme_run",
            },
            {
                "name": "CEME+",
                "uses_known_w_sd": True,
                "run_type": "education_wage_ceme_run",
            },
            {
                "name": "Oracle",
                "uses_t_instead_of_w": True,
                "run_type": "education_wage_mlp_run",
            },
            {
                "name": "Naive",
                "uses_t_instead_of_w": False,
                "run_type": "education_wage_mlp_run",
            },
        ],
        "me_magnitudes": [
            {
                "name": "0%",
                "relative_w_sd": 0.0,
            },
            {
                "name": "20%",
                "relative_w_sd": 0.2,
            },
            {
                "name": "40%",
                "relative_w_sd": 0.4,
            },
            {
                "name": "60%",
                "relative_w_sd": 0.6,
            },
            {
                "name": "80%",
                "relative_w_sd": 0.8,
            },
            {
                "name": "100%",
                "relative_w_sd": 1.0,
            },
        ],
    }
    return param_grid
