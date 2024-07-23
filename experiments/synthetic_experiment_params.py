from typing import Dict, List, Any


def get_constant_params() -> Dict[str, Any]:
    constant_params: dict[str, Any] = {
        "gp_data_generator_params": {
            "n_total": 44000,
            "z_mean": 0,
            "z_sd": 1,
            "kernel_alpha": 1,
            "kernel_scale": 2,
            "n_grid_points": 1000,
        },
        "uses_dreg": False,
        "n_validate": 8000,
        "n_test": 20000,
        "max_epochs": 500,
        "encoder_hidden_sizes": [20, 20, 20],
        "decoder_hidden_sizes": [20, 20, 20],
        "weight_decay": 0,
        "n_importance_samples": 32,
        "n_likelihood_samples": 2048,
        "num_q_annealing_epochs": 10,
        "q_initial_weight": 4,
        "lr_reducer_patience": 30,
        "lr_reducer_factor": 0.1,
        "patience_in_epochs": 40,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.97,
        "max_training_time_s": 1560,
        "optimizer": "adam",
        "sample_size_for_aid_integral": 10000,
        "base_seed": 0,
    }
    return constant_params


def get_param_grid() -> Dict[str, List[Dict[str, Any]]]:
    noise_magnitude_names = ["small", "medium", "large"]
    relative_w_sds = [0.1, 0.2, 0.4]
    relative_y_sds = [0.1, 0.2, 0.4]
    param_grid: Dict[str, List[Dict[str, Any]]] = {
        "noise_magnitude_and_random_seed_for_data_generation": [
            {
                "name": f"{noise_magnitude_names[seed % 3]},{seed}",
                "gp_data_generator_params": {
                    "relative_w_sd": relative_w_sds[seed % 3],
                    "relative_y_sd": relative_y_sds[seed % 3],
                    "local_seed": seed,
                },
            }
            for seed in range(201)
        ],
        "dummy_axis_to_get_repetitions": [{"name": i} for i in range(6)],
        "algorithm_and_n_train": [
            {
                "name": "CEME,1000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_ceme_run",
                "batch_size": 64,
                "learning_rate": 0.003,
                "n_train": 1000,
            },
            {
                "name": "CEME,4000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_ceme_run",
                "batch_size": 64,
                "learning_rate": 0.003,
                "n_train": 4000,
            },
            {
                "name": "CEME,16000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_ceme_run",
                "batch_size": 256,
                "learning_rate": 0.01,
                "n_train": 16000,
            },
            {
                "name": "CEME+,1000",
                "uses_known_w_sd": True,
                "run_type": "synthetic_ceme_run",
                "batch_size": 64,
                "learning_rate": 0.003,
                "n_train": 1000,
            },
            {
                "name": "CEME+,4000",
                "uses_known_w_sd": True,
                "run_type": "synthetic_ceme_run",
                "batch_size": 64,
                "learning_rate": 0.003,
                "n_train": 4000,
            },
            {
                "name": "CEME+,16000",
                "uses_known_w_sd": True,
                "run_type": "synthetic_ceme_run",
                "batch_size": 256,
                "learning_rate": 0.01,
                "n_train": 16000,
            },
            {
                "name": "Oracle,1000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_mlp_run",
                "batch_size": 64,
                "learning_rate": 0.001,
                "n_train": 1000,
                "uses_t_instead_of_w": True,
            },
            {
                "name": "Oracle,4000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_mlp_run",
                "batch_size": 64,
                "learning_rate": 0.001,
                "n_train": 4000,
                "uses_t_instead_of_w": True,
            },
            {
                "name": "Oracle,16000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_mlp_run",
                "batch_size": 64,
                "learning_rate": 0.001,
                "n_train": 16000,
                "uses_t_instead_of_w": True,
            },
            {
                "name": "Naive,1000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_mlp_run",
                "batch_size": 64,
                "learning_rate": 0.001,
                "n_train": 1000,
                "uses_t_instead_of_w": False,
            },
            {
                "name": "Naive,4000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_mlp_run",
                "batch_size": 64,
                "learning_rate": 0.001,
                "n_train": 4000,
                "uses_t_instead_of_w": False,
            },
            {
                "name": "Naive,16000",
                "uses_known_w_sd": False,
                "run_type": "synthetic_mlp_run",
                "batch_size": 64,
                "learning_rate": 0.001,
                "n_train": 16000,
                "uses_t_instead_of_w": False,
            },
        ],
    }

    return param_grid
