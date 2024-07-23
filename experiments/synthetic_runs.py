import git
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import data.gp_data_generator as gp_data_generator
import experiments.run_common as run_common
import models.model_ceme as model_ceme
import models.model_fitting as model_fitting
import models.model_mlp as model_mlp
import utils


def synthetic_ceme_run(params):
    print(f"Running synthetic_ceme_run with params: {params}")

    output = {}

    time_start_s: float = time.monotonic()

    data_generator = gp_data_generator.GpDataGenerator(
        **params["gp_data_generator_params"]
    )

    train_data = data_generator.generate_data(params["n_train"])
    validate_data = data_generator.generate_data(params["n_validate"])
    test_data = data_generator.generate_data(params["n_test"])
    train_dataloader = DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    if params["uses_known_w_sd"]:
        known_w_sd = data_generator.get_w_sd()
    else:
        known_w_sd = None

    output["time_data_generated_s"] = time.monotonic() - time_start_s

    time_start_s = time.monotonic()

    model = model_ceme.Ceme(
        z_dim=train_data.z.shape[1],
        n_importance_samples=params["n_importance_samples"],
        n_likelihood_samples=params["n_likelihood_samples"],
        encoder_hidden_sizes=params["encoder_hidden_sizes"],
        decoder_hidden_sizes=params["decoder_hidden_sizes"],
        known_w_sd=known_w_sd,
    )

    output["time_model_created_s"] = time.monotonic() - time_start_s
    time_start_s = time.monotonic()

    output["was_run_interrupted_due_to_time"] = model_fitting.fit_model(
        model=model,
        train_dataloader=train_dataloader,
        validate_data=validate_data,
        learning_rate=params["learning_rate"],
        max_epocs=params["max_epochs"],
        weight_decay=params["weight_decay"],
        num_q_annealing_epochs=params["num_q_annealing_epochs"],
        q_initial_weight=params["q_initial_weight"],
        lr_reducer_patience=params["lr_reducer_patience"],
        lr_reducer_factor=params["lr_reducer_factor"],
        patience_in_epochs=params["patience_in_epochs"],
        adam_beta_1=params["adam_beta_1"],
        adam_beta_2=params["adam_beta_2"],
        max_training_time_s=params["max_training_time_s"],
    )

    output["time_model_trained_s"] = time.monotonic() - time_start_s

    repo = git.Repo(search_parent_directories=True)
    output["git_commit_hash"] = repo.head.object.hexsha

    output["params"] = params

    (
        output["y_idrf_error"],
        output["t_idrf_error"],
        output["t_sd_idrf_error"],
    ) = run_common.get_idrf_errors_ceme(
        test_data,
        model,
        data_generator,
    )

    output["train_likelihood"] = run_common.get_likelihood_on(train_data, model)
    output["validate_likelihood"] = run_common.get_likelihood_on(validate_data, model)
    output["test_likelihood"] = run_common.get_likelihood_on(test_data, model)

    output["w_sd_estimate"] = model.decoder.w_sd.item()
    output["y_sd_estimate"] = model.decoder.y_sd.item()
    output["absolute_w_sd_error"] = np.float64(
        np.abs(model.decoder.w_sd.item() - data_generator.get_w_sd())
    )
    output["relative_w_sd_error"] = np.float64(
        output["absolute_w_sd_error"] / data_generator.get_w_sd()
    )
    output["absolute_y_sd_error"] = np.float64(
        np.abs(output["y_sd_estimate"] - data_generator.get_y_sd())
    )
    output["relative_y_sd_error"] = np.float64(
        output["absolute_y_sd_error"] / data_generator.get_y_sd()
    )
    output["w_sd_gt"] = data_generator.get_w_sd()
    output["y_sd_gt"] = data_generator.get_y_sd()

    y_mean = test_data.y.mean()
    y_sd = test_data.y.std()
    y_min = y_mean - 4 * y_sd
    y_max = y_mean + 4 * y_sd

    def y_mean_func_estimate(z, t):
        return (
            model.decoder.y_mean_estimate(
                torch.from_numpy(z.copy()), torch.from_numpy(t.copy())
            )
            .detach()
            .numpy()
        )

    start_t = time.monotonic()

    (
        output["aid_estimate"],
        output["aid_estimate_variance"],
    ) = run_common.get_average_interventional_distance(
        t_sample=test_data.t[0 : params["sample_size_for_aid_integral"]],
        y_min=y_min,
        y_max=y_max,
        y_mean_func_estimate=y_mean_func_estimate,
        y_sd_estimate=output["y_sd_estimate"],
        y_mean_func_gt=data_generator.y_mean_func,
        y_sd_gt=output["y_sd_gt"],
        z_mean_gt=data_generator.z_mean,
        z_sd_gt=data_generator.z_sd,
        z_train_and_validate=np.concatenate((train_data.z, validate_data.z), axis=0),
        n_z_gt=100,
    )

    output["aid_estimation_time_s"] = time.time() - start_t

    output = utils.get_json_serializable_object(output)

    print("Output:")
    print(json.dumps(output, sort_keys=True, indent=4))

    return output


def synthetic_mlp_run(params):
    print(f"Running synthetic_mlp_run with params: {params}")

    output = {}

    time_start_s: float = time.monotonic()

    data_generator = gp_data_generator.GpDataGenerator(
        **params["gp_data_generator_params"]
    )

    train_data = data_generator.generate_data(params["n_train"])
    validate_data = data_generator.generate_data(params["n_validate"])
    test_data = data_generator.generate_data(params["n_test"])
    train_dataloader = DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )

    output["time_data_generated_s"] = time.monotonic() - time_start_s

    time_start_s = time.monotonic()

    model = model_mlp.Mlp(
        z_dim=train_data.z.shape[1],
        hidden_sizes=params["decoder_hidden_sizes"],
        uses_t_instead_of_w=params["uses_t_instead_of_w"],
    )

    output["time_model_created_s"] = time.monotonic() - time_start_s
    time_start_s = time.monotonic()

    output["was_run_interrupted_due_to_time"] = model_fitting.fit_model(
        model=model,
        train_dataloader=train_dataloader,
        validate_data=validate_data,
        learning_rate=params["learning_rate"],
        max_epocs=params["max_epochs"],
        weight_decay=params["weight_decay"],
        num_q_annealing_epochs=params["num_q_annealing_epochs"],
        q_initial_weight=params["q_initial_weight"],
        lr_reducer_patience=params["lr_reducer_patience"],
        lr_reducer_factor=params["lr_reducer_factor"],
        patience_in_epochs=params["patience_in_epochs"],
        adam_beta_1=params["adam_beta_1"],
        adam_beta_2=params["adam_beta_2"],
        max_training_time_s=params["max_training_time_s"],
    )

    output["time_model_trained_s"] = time.monotonic() - time_start_s

    repo = git.Repo(search_parent_directories=True)
    output["git_commit_hash"] = repo.head.object.hexsha

    output["params"] = params

    output["y_idrf_error"] = run_common.get_y_idrf_error_mlp(
        test_data,
        model,
        data_generator,
    )

    output["w_sd_estimate"] = None
    output["y_sd_estimate"] = run_common.get_y_sd_estimate_mlp(
        model=model,
        test_data=test_data,
        uses_t_instead_of_w=params["uses_t_instead_of_w"],
    )

    output["absolute_w_sd_error"] = None
    output["relative_w_sd_error"] = None

    output["train_loss"] = (
        model.loss(
            batch=[
                torch.tensor(train_data.z),
                torch.tensor(train_data.t),
                torch.tensor(train_data.w),
                torch.tensor(train_data.y),
                None,
            ]
        )
        .detach()
        .item()
    )
    output["validate_loss"] = (
        model.loss(
            batch=[
                torch.tensor(validate_data.z),
                torch.tensor(validate_data.t),
                torch.tensor(validate_data.w),
                torch.tensor(validate_data.y),
                None,
            ]
        )
        .detach()
        .item()
    )
    output["test_loss"] = (
        model.loss(
            batch=[
                torch.tensor(test_data.z),
                torch.tensor(test_data.t),
                torch.tensor(test_data.w),
                torch.tensor(test_data.y),
                None,
            ]
        )
        .detach()
        .item()
    )

    output["absolute_y_sd_error"] = np.float64(
        np.abs(output["y_sd_estimate"] - data_generator.get_y_sd())
    )
    output["relative_y_sd_error"] = np.float64(
        output["absolute_y_sd_error"] / data_generator.get_y_sd()
    )
    output["w_sd_gt"] = data_generator.get_w_sd()
    output["y_sd_gt"] = data_generator.get_y_sd()

    y_mean = test_data.y.mean()
    y_sd = test_data.y.std()
    y_min = y_mean - 4 * y_sd
    y_max = y_mean + 4 * y_sd

    def y_mean_func_estimate(z, t):
        with torch.no_grad():
            return model(torch.from_numpy(z), torch.from_numpy(t)).numpy()

    start_t = time.monotonic()

    (
        output["aid_estimate"],
        output["aid_estimate_variance"],
    ) = run_common.get_average_interventional_distance(
        t_sample=test_data.t[0 : params["sample_size_for_aid_integral"]],
        y_min=y_min,
        y_max=y_max,
        y_mean_func_estimate=y_mean_func_estimate,
        y_sd_estimate=output["y_sd_estimate"],
        y_mean_func_gt=data_generator.y_mean_func,
        y_sd_gt=output["y_sd_gt"],
        z_mean_gt=data_generator.z_mean,
        z_sd_gt=data_generator.z_sd,
        z_train_and_validate=np.concatenate((train_data.z, validate_data.z), axis=0),
        n_z_gt=100,
    )

    output["aid_estimation_time_s"] = time.time() - start_t

    output = utils.get_json_serializable_object(output)

    print("Output:")
    print(json.dumps(output, sort_keys=True, indent=4))

    return output
