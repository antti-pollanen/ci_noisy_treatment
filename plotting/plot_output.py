import itertools

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns


def add_features(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df["rel_w_sd_diff"] = (
        data_df["w_sd_estimate"] - data_df["w_sd_gt"]
    ) / data_df["w_sd_gt"]
    data_df["rel_y_sd_diff"] = (
        data_df["y_sd_estimate"] - data_df["y_sd_gt"]
    ) / data_df["y_sd_gt"]

    if "params.param_grid.algorithm_and_n_train.name" in data_df.columns:
        data_df["algorithm"] = data_df[
            "params.param_grid.algorithm_and_n_train.name"
        ].apply(lambda s: s.split(",")[0])
        data_df["training_set_size"] = data_df[
            "params.param_grid.algorithm_and_n_train.name"
        ].apply(lambda s: s.split(",")[1])

    if (
        "params.param_grid.noise_magnitude_and_random_seed_for_data_generation.name"
        in data_df.columns
    ):
        data_df["noise_magnitude"] = data_df[
            "params.param_grid.noise_magnitude_and_random_seed_for_data_generation.name"
        ].apply(lambda s: s.split(",")[0])
        data_df["random_seed_for_data_generation"] = data_df[
            "params.param_grid.noise_magnitude_and_random_seed_for_data_generation.name"
        ].apply(lambda s: s.split(",")[1])

    try:
        data_df["test_score"] = data_df.apply(
            lambda row: (
                row["test_likelihood"]
                if row["algorithm"] in ["CEME", "CEME+"]
                else -row["test_loss"]
            ),
            axis=1,
        )
    except KeyError:
        print(
            "no test_score column added due to missing test_likelihood, test_loss or algorithm column)"
        )

    try:
        data_df["validate_score"] = data_df.apply(
            lambda row: (
                row["validate_likelihood"]
                if row["algorithm"] in ["CEME", "CEME+"]
                else -row["validate_loss"]
            ),
            axis=1,
        )
    except KeyError:
        print(
            "no validate_score column added due to missing validate_likelihood, validate_loss or algorithm column"
        )
    return data_df


def get_best_run_indices_per_run_type(
    data_df: pd.DataFrame, num_runs_to_take_per_type: int
) -> list[int]:
    assert np.issubdtype(data_df.dtypes["validate_score"], np.number)

    # find the mapping from parameter combination to the list of indices corresponding
    # to the runs that have that combination

    algs = data_df.algorithm.unique()
    training_set_sizes = data_df.training_set_size.unique()
    noise_magnitudes = data_df.noise_magnitude.unique()
    dataset_indices = data_df.random_seed_for_data_generation.unique()

    params_to_indices = {}
    for alg in algs:
        for training_set_size in training_set_sizes:
            for noise_magnitude in noise_magnitudes:
                for dataset_index in dataset_indices:
                    params_to_indices[
                        (alg, training_set_size, noise_magnitude, dataset_index)
                    ] = []

    print("computing params_to_indices")
    for index, row in data_df.iterrows():
        params_to_indices[
            (
                row["algorithm"],
                row["training_set_size"],
                row["noise_magnitude"],
                row["random_seed_for_data_generation"],
            )
        ].append(index)

    # sort the indices in order of validate score
    print("sorting indices")
    test_score_for_sorting = lambda i: data_df.iloc[i]["validate_score"]
    for key in params_to_indices:
        params_to_indices[key].sort(key=test_score_for_sorting, reverse=True)

    # discard runs so that we have the n best left for each type
    print("discarding runs")
    for params, indices in params_to_indices.items():
        params_to_indices[params] = indices[0:num_runs_to_take_per_type]

    # aggregate the indices into one list
    print("aggregating the indices to pick to one list")
    # all_indices = []
    # for _, indices in params_to_indices.items():
    #     all_indices = all_indices + indices

    all_indices = list(itertools.chain.from_iterable(params_to_indices.values()))

    return all_indices


tidied_label_for_data_label_synthetic: dict[str, str] = {
    "noise_magnitude": "noise level",
    "training_set_size": "training set size",
    "test_likelihood": "test likelihood",
    "test_loss": "test loss",
    "y_idrf_error": r"$\sqrt{\mathrm{MSE}}$",
    "t_idrf_error": r"$\sqrt{\mathrm{MSE}}$",
    "t_sd_idrf_error": r"$\sqrt{\mathrm{MSE}}$",
    "rel_w_sd_diff": "relative error",
    "rel_y_sd_diff": "relative error",
    "model_building_and_fitting_time_minutes": "time (minutes)",
    "CEME": "CEME",
    "CEME+": r"CEME$^+$",
    "Oracle": "Oracle",
    "Naive": "Naive",
    "1000": "1000",
    "4000": "4000",
    "16000": "16000",
    "aid_estimate": "AID estimate",
}


def tidify_labels(ax: plt.Axes, tidied_label_for_data_label: dict[str, str]) -> None:
    ax.set_xlabel(tidied_label_for_data_label[ax.get_xlabel()])
    ax.set_ylabel(tidied_label_for_data_label[ax.get_ylabel()])

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        legend_handles,
        [tidied_label_for_data_label[label] for label in legend_labels],
        loc="best",
        ncol=1,
    )


def plot_y_cond_mean_errors_synthetic(
    data_df_pruned: pd.DataFrame,
) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16.3, 6))
    noise_levels: list[str] = ["small", "medium", "large"]
    noise_level_labels: list[str] = ["10%", "20%", "40%"]

    for i in range(3):
        sns.boxplot(
            x="training_set_size",
            y="y_idrf_error",
            hue="algorithm",
            data=data_df_pruned[data_df_pruned["noise_magnitude"] == noise_levels[i]],
            ax=axes[i],
        )
        axes[i].set_title("noise level: " + noise_level_labels[i])
        axes[i].set_ylim(0, 0.265)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        [tidied_label_for_data_label_synthetic[label] for label in legend_labels],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
    )

    for ax in axes:
        tidify_labels(ax, tidied_label_for_data_label_synthetic)
        ax.get_legend().remove()
        ax.yaxis.set_tick_params(pad=-1.0)

    fig.suptitle(r"Error in $\mathrm{E}[y|z,do(x^*)]$ estimation", y=1.03)

    return fig


def plot_y_noise_errors_synthetic(
    data_df_pruned: pd.DataFrame,
) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16.3, 6))
    noise_levels: list[str] = ["small", "medium", "large"]
    noise_level_labels: list[str] = ["10%", "20%", "40%"]

    for i in range(3):
        sns.boxplot(
            x="training_set_size",
            y="rel_y_sd_diff",
            hue="algorithm",
            data=data_df_pruned[data_df_pruned["noise_magnitude"] == noise_levels[i]],
            ax=axes[i],
        )
        axes[i].set_title("noise level: " + noise_level_labels[i])

        axes[i].yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
        )
        axes[i].set_ylim(-0.1, 1.0)

    xmin = -0.52
    xmax = 2.52
    gt_lines = []
    for ax in axes:
        gt_lines.append(
            ax.hlines(0, xmin=xmin, xmax=xmax, linestyles="dashed", colors="black")
        )
        ax.set_xlim(xmin, xmax)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        legend_handles + [gt_lines[0]],
        [tidied_label_for_data_label_synthetic[label] for label in legend_labels]
        + ["ground truth"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
    )

    for ax in axes:
        ax.set_xlabel(tidied_label_for_data_label_synthetic[ax.get_xlabel()])
        ax.set_ylabel(tidied_label_for_data_label_synthetic[ax.get_ylabel()])
        ax.get_legend().remove()
        ax.yaxis.set_tick_params(pad=-1.0)

    fig.suptitle(r"Estimation of $\Delta Y$ standard deviation", y=1.03)

    return fig


def plot_aid_estimates_synthetic(
    data_df_pruned: pd.DataFrame,
) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16.3, 6))
    noise_levels: list[str] = ["small", "medium", "large"]
    noise_level_labels: list[str] = ["10%", "20%", "40%"]

    for i in range(3):
        sns.boxplot(
            x="training_set_size",
            y="aid_estimate",
            hue="algorithm",
            data=data_df_pruned[data_df_pruned["noise_magnitude"] == noise_levels[i]],
            ax=axes[i],
        )
        axes[i].set_title("noise level: " + noise_level_labels[i])
        axes[i].set_ylim(0, 0.421)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        [tidied_label_for_data_label_synthetic[label] for label in legend_labels],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
    )

    for ax in axes:
        tidify_labels(ax, tidied_label_for_data_label_synthetic)
        ax.get_legend().remove()

        ax.yaxis.set_tick_params(pad=-1.0)

    fig.suptitle(r"Error in $p(y|do(x^*))$ estimation", y=1.03)

    return fig


def plot_x_noise_errors_synthetic(
    data_df_pruned: pd.DataFrame,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot()

    sns.boxplot(
        x="noise_magnitude",
        y="rel_w_sd_diff",
        hue="training_set_size",
        data=data_df_pruned[data_df_pruned["algorithm"] == "CEME"],
        ax=ax,
    )

    xmin = -0.52
    xmax = 2.52
    gt_line = ax.hlines(0, xmin=xmin, xmax=xmax, linestyles="dashed", colors="black")  # type: ignore
    ax.set_xlim(xmin, xmax)

    ax.set_xticklabels(["10%", "20%", "40%"])

    ax.set_ylim(-0.71, 0.45)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

    ax.set_xlabel(tidied_label_for_data_label_synthetic[ax.get_xlabel()])
    ax.set_ylabel(tidied_label_for_data_label_synthetic[ax.get_ylabel()])

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    first_legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        title="training set size",
    )
    plt.gca().add_artist(first_legend)

    ax.legend(
        [gt_line],
        ["ground truth"],
        loc="upper right",
    )

    ax.set_title(r"Estimation of $\Delta X$ standard deviation")

    ax.yaxis.set_tick_params(pad=-1.0)

    return fig


tidied_label_for_data_label_education_wage: dict[str, str] = {
    "algorithm": "algorithm",
    "params.param_grid.algorithm.name": "algorithm",
    "noise_magnitude": "noise level",
    "me_magnitudes": r"relative SD of $\Delta X$",
    "params.param_grid.me_magnitudes.name": r"relative SD of $\Delta X$",
    "training_set_size": "training set size",
    "test_likelihood": "test likelihood",
    "test_loss": "test loss",
    "y_idrf_error": r"$\sqrt{\mathrm{MSE}}$",
    "t_idrf_error": r"$\sqrt{\mathrm{MSE}}$",
    "t_sd_idrf_error": r"$\sqrt{\mathrm{MSE}}$",
    "y_idrf_uniform_error": "y_idrf_uniform_error",
    "rel_w_sd_diff": "relative error",
    "rel_y_sd_diff": "relative error",
    "w_sd_estimate": r"$\Delta X$ SD estimate",
    "y_sd_estimate": r"$\Delta Y$ SD estimate",
    "model_building_and_fitting_time_minutes": "time (minutes)",
    "CEME": "CEME",
    "CEME+": r"CEME$^+$",
    "Oracle": "Oracle",
    "Naive": "Naive",
    "1000": "1000",
    "4000": "4000",
    "16000": "16000",
    "0%": "0%",
    "20%": "20%",
    "40%": "40%",
    "60%": "60%",
    "80%": "80%",
    "100%": "100%",
    "400%": "400%",
    "ground truth": "ground truth",
    "aid_estimate": "AID estimate",
}


def plot_y_cond_mean_errors_education_wage(
    data_df: pd.DataFrame,
    title: str = r"Error in $\mathrm{E}[y|z,do(x^*)]$ estimation",
    x: str = "params.param_grid.me_magnitudes.name",
    y: str = "y_idrf_error",
    hue: str = "params.param_grid.algorithm.name",
):
    fig = plt.figure(figsize=(8, 6))

    ax = sns.lineplot(
        estimator="median",
        x=x,
        y=y,
        hue=hue,
        hue_order=["CEME", "CEME+", "Oracle", "Naive"],
        errorbar=("pi", 50),
        err_style="band",
        data=data_df,
        marker="o",
        markersize=8,
        ax=None,
    )
    ax.set_title(title)

    tidify_labels(ax, tidied_label_for_data_label_education_wage)
    return fig


def plot_y_noise_errors_education_wage(
    data_df: pd.DataFrame,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(8, 6))

    ax = sns.lineplot(
        estimator="median",
        x="params.param_grid.me_magnitudes.name",
        y="y_sd_estimate",
        hue="params.param_grid.algorithm.name",
        hue_order=["CEME", "CEME+", "Oracle", "Naive"],
        errorbar=("pi", 50),
        err_style="band",
        data=data_df,
        marker="o",
        markersize=8,
    )

    ax = sns.lineplot(
        x="params.param_grid.me_magnitudes.name",
        y="y_sd_gt",
        data=data_df,
        dashes=True,
        ax=ax,
    )

    ax.lines[-1].set_color("black")
    ax.lines[-1].set_linestyle("--")

    tidify_labels(ax, tidied_label_for_data_label_education_wage)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        legend_handles + [ax.lines[-1]],
        [tidied_label_for_data_label_education_wage[label] for label in legend_labels]
        + ["ground truth"],
        loc="best",
        ncol=1,
    )
    ax.set_title(r"Estimation of $\Delta Y$ standard deviation")

    return fig


def plot_aid_estimates_education_wage(
    data_df: pd.DataFrame,
    title: str = r"Error in $p(y|do(x^*))$ estimation",
    x: str = "params.param_grid.me_magnitudes.name",
    y: str = "aid_estimate",
    hue: str = "params.param_grid.algorithm.name",
    estimator: str = "median",
    err_style: str = "band",
    errorbar: tuple[str, int] = ("pi", 50),
) -> matplotlib.figure.Figure:

    fig = plt.figure(figsize=(8, 6))

    err_kws = {}
    markersize = 8
    if err_style == "bars":
        err_kws = {"capsize": 6, "capthick": 1.4}
        markersize = 7

    ax = sns.lineplot(
        estimator=estimator,
        x=x,
        y=y,
        hue=hue,
        hue_order=["CEME", "CEME+", "Oracle", "Naive"],
        errorbar=errorbar,
        err_style=err_style,
        err_kws=err_kws,
        data=data_df,
        marker="o",
        markersize=markersize,
        ax=None,
    )
    ax.set_title(title)

    tidify_labels(ax, tidied_label_for_data_label_education_wage)

    return fig


def plot_x_noise_errors_education_wage(
    data_df: pd.DataFrame,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(8, 6))

    data_df = data_df.loc[data_df["params.param_grid.algorithm.name"] == "CEME"]

    ax = sns.lineplot(
        estimator="median",
        x="params.param_grid.me_magnitudes.name",
        y="w_sd_estimate",
        hue="params.param_grid.algorithm.name",
        errorbar=("pi", 50),
        err_style="band",
        data=data_df,
        marker="o",
        markersize=8,
    )

    data_df_gt = data_df.copy()
    data_df_gt["ground_truth_name"] = "ground truth"

    ax = sns.lineplot(
        x="params.param_grid.me_magnitudes.name",
        y="w_sd_gt",
        data=data_df_gt,
        dashes=True,
        ax=ax,
    )

    ax.lines[-1].set_color("black")
    ax.lines[-1].set_linestyle("--")

    tidify_labels(ax, tidied_label_for_data_label_education_wage)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        legend_handles + [ax.lines[-1]],
        [tidied_label_for_data_label_education_wage[label] for label in legend_labels]
        + ["ground truth"],
        loc="best",
        ncol=1,
    )
    ax.set_title(r"Estimation of $\Delta X$ standard deviation")

    return fig
