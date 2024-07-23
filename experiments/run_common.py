import numpy as np
import scipy
import torch

import data.data_utils as data_utils
import models.model_ceme as model_ceme
import models.model_mlp as model_mlp


def get_idrf_errors_ceme(
    dataset,
    model_continuous,
    data_generator,
    errors=["y_mean_error", "t_mean_error", "t_sd_error"],
):
    assert len(dataset.z.shape) == 2, dataset.z.shape
    assert len(dataset.t.shape) == 2, dataset.t.shape
    assert len(dataset.y.shape) == 2, dataset.y.shape

    z = torch.from_numpy(dataset.z)
    t = torch.from_numpy(dataset.t)

    y_mean_error = None
    t_mean_error = None
    t_sd_error = None

    if "y_mean_error" in errors:
        y_mean_estimate = (
            model_continuous.decoder.y_mean_estimate(z, t).detach().numpy()
        )

        assert dataset.y.shape == y_mean_estimate.shape, (
            dataset.y.shape + y_mean_estimate.shape
        )

        y_mean_gt = data_generator.y_mean_func(dataset.z, dataset.t).reshape(-1, 1)
        assert dataset.y.shape == y_mean_gt.shape, dataset.y.shape + y_mean_gt.shape
        y_mean_error = np.sqrt(np.square(y_mean_gt - y_mean_estimate).mean())

    if "t_mean_error" in errors:
        t_mean_estimate = model_continuous.decoder.t_mean_estimate(z).detach().numpy()

        assert dataset.t.shape == t_mean_estimate.shape, (
            dataset.t.shape + t_mean_estimate.shape
        )
        t_mean_gt = data_generator.t_mean_func(dataset.z).reshape(-1, 1)
        t_mean_error = np.sqrt(np.square(t_mean_gt - t_mean_estimate).mean())

    if "t_sd_error" in errors:
        t_sd_estimate = model_continuous.decoder.t_sd_estimate(z).detach().numpy()

        assert dataset.t.shape == t_sd_estimate.shape, (
            dataset.t.shape + t_sd_estimate.shape
        )
        t_sd_gt = data_generator.t_sd_func(dataset.z).reshape(-1, 1)
        t_sd_error = np.sqrt(np.square(t_sd_gt - t_sd_estimate).mean())

    return y_mean_error, t_mean_error, t_sd_error


def get_likelihood_on(data: data_utils.MeDataset, model: model_ceme.Ceme):
    return model.likelihood(
        torch.from_numpy(data.z),
        torch.from_numpy(data.w),
        torch.from_numpy(data.y),
    ).item()


def get_y_do_t_density(
    y, t, y_mean_func, y_sd, z_for_expectation, max_array_size=10**8
):
    # calculates the p(y|do(t)) density for parallel arrays of y and t

    assert len(z_for_expectation.shape) in [1, 2], z_for_expectation.shape
    assert len(t.shape) == 1 or (len(t.shape) == 2 and t.shape[1] == 1), t.shape
    assert y.shape == t.shape, "y.shape: " + str(y.shape) + " t.shape: " + str(t.shape)

    t = t.view()
    y = y.view()
    z_for_expectation = z_for_expectation.view()

    n_t = t.shape[0]
    n_z = z_for_expectation.shape[0]

    t.shape = (n_t, 1)
    y.shape = (n_t, 1)
    z_for_expectation.shape = (n_z, -1)

    num_z_features = z_for_expectation.shape[1]
    num_t_features = 1
    num_y_features = 1

    ret = np.zeros(shape=n_t)
    n_t_chunk = max_array_size // (n_z * max(num_t_features, num_z_features))

    for i in range(0, n_t, n_t_chunk):
        print(
            "Calculating y_do_t density in chunks for chunk number ",
            i // n_t_chunk,
            " / ",
            t.shape[0] // n_t_chunk,
        )

        n_t_current_chunk = min(n_t_chunk, n_t - i)

        t_chunk = t[i : i + n_t_current_chunk]
        y_chunk = y[i : i + n_t_current_chunk]
        assert not t_chunk.base is None
        assert not y_chunk.base is None

        t_reshaped = t_chunk.reshape(n_t_current_chunk, 1, num_t_features)
        assert not t_reshaped.base is None
        z_reshaped = z_for_expectation.reshape(1, n_z, num_z_features)
        assert not z_reshaped.base is None

        t_grid = np.broadcast_to(t_reshaped, (n_t_current_chunk, n_z, num_t_features))
        assert not t_grid.base is None

        z_grid = np.broadcast_to(z_reshaped, (n_t_current_chunk, n_z, num_z_features))
        assert not z_grid.base is None

        # creates copies, complicated to avoid
        z_grid_as_vector = z_grid.reshape(-1, num_z_features)
        t_grid_as_vector = t_grid.reshape(-1, num_t_features)

        y_means = y_mean_func(z_grid_as_vector, t_grid_as_vector)
        y_means.shape = (n_t_current_chunk, n_z, num_y_features)

        y_for_pdf_computation = y_chunk.view()
        y_for_pdf_computation.shape = (n_t_current_chunk, 1, num_y_features)
        densities = scipy.stats.norm(y_means, y_sd).pdf(y_for_pdf_computation)

        ret_chunk = densities.mean(axis=1)
        ret_chunk.shape = n_t_current_chunk

        ret[i : i + n_t_current_chunk] = ret_chunk
    return ret


def get_y_do_t_density_numeric_integration(
    y,
    t,
    y_mean_func,
    y_sd,
    z_mean: float,
    z_sd: float,
    n_z: int,
    max_array_size: int = 10**8,
):
    # Calculates the p(y|do(t)) density for parallel arrays of y and t
    # Assumes univariate Gaussian z
    assert len(t.shape) == 1 or (len(t.shape) == 2 and t.shape[1] == 1), t.shape
    assert y.shape == t.shape, "y.shape: " + str(y.shape) + " t.shape: " + str(t.shape)

    t = t.view()
    y = y.view()
    t.shape = (-1, 1)
    y.shape = (-1, 1)

    n_t = t.shape[0]

    num_t_features = 1
    num_z_features = 1
    num_y_features = 1

    z_min = z_mean - 3 * z_sd
    z_max = z_mean + 3 * z_sd

    z_linspace = np.linspace(z_min, z_max, n_z).reshape(n_z, 1)
    z_densities = scipy.stats.norm(z_mean, z_sd).pdf(z_linspace)
    assert z_densities.shape == (n_z, 1)

    ret = np.zeros(shape=n_t)
    n_t_chunk = max_array_size // (n_z * max(num_t_features, num_z_features))

    # integrate z_density over z_linspace using numpy trapz
    z_densities_integral = np.trapz(y=z_densities, x=z_linspace, axis=0)
    assert z_densities_integral.shape == (1,), z_densities_integral.shape

    for i in range(0, n_t, n_t_chunk):
        print(
            "Calculating y_do_t density in chunks for chunk number ",
            i // n_t_chunk,
            " / ",
            t.shape[0] // n_t_chunk,
        )

        n_t_current_chunk = min(n_t_chunk, n_t - i)

        t_chunk = t[i : i + n_t_current_chunk]
        y_chunk = y[i : i + n_t_current_chunk]
        assert not t_chunk.base is None
        assert not y_chunk.base is None

        t_reshaped = t_chunk.reshape(n_t_current_chunk, 1, num_t_features)
        assert not t_reshaped.base is None
        z_reshaped = z_linspace.reshape(1, n_z, num_z_features)
        assert not z_reshaped.base is None

        t_grid = np.broadcast_to(t_reshaped, (n_t_current_chunk, n_z, num_t_features))
        assert not t_grid.base is None

        z_grid = np.broadcast_to(z_reshaped, (n_t_current_chunk, n_z, num_z_features))
        assert not z_grid.base is None

        z_grid_as_vector = z_grid.reshape(-1, num_z_features)
        t_grid_as_vector = t_grid.reshape(-1, num_t_features)

        y_means = y_mean_func(z_grid_as_vector, t_grid_as_vector)
        y_means.shape = (n_t_current_chunk, n_z, num_y_features)

        y_for_pdf_computation = y_chunk.view()
        y_for_pdf_computation.shape = (n_t_current_chunk, 1, num_y_features)
        y_densities = scipy.stats.norm(y_means, y_sd).pdf(y_for_pdf_computation)

        assert y_densities.shape == (n_t_current_chunk, n_z, 1), (
            y_densities.shape,
            (n_t_current_chunk, n_z, 1),
        )

        integrand = y_densities * z_densities

        assert integrand.shape == (n_t_current_chunk, n_z, 1)

        ret_chunk = (
            np.trapz(y=integrand, x=z_grid, axis=1).reshape(n_t_current_chunk)
            / z_densities_integral
        )

        ret[i : i + n_t_current_chunk] = ret_chunk

    return ret


def get_average_interventional_distance(
    t_sample,
    y_min: float,
    y_max: float,
    y_mean_func_estimate,
    y_sd_estimate,
    y_mean_func_gt,
    y_sd_gt,
    z_mean_gt,
    z_sd_gt,
    z_train_and_validate,
    n_z_gt=100,
):
    print("Computing average interventional distance")

    t_sample = t_sample.view()

    assert len(t_sample.shape) == 1 or (
        len(t_sample.shape) == 2 and t_sample.shape[1] == 1
    ), t_sample.shape

    t_sample.shape = (-1, 1)

    y_sample = np.random.uniform(y_min, y_max, size=t_sample.shape)

    print("Computing p(y|do(t)) estimates")
    densities_estimate = get_y_do_t_density(
        y=y_sample,
        t=t_sample,
        y_mean_func=y_mean_func_estimate,
        y_sd=y_sd_estimate,
        z_for_expectation=z_train_and_validate,
    )

    print("Computing p(y|do(t)) ground truths")
    densities_gt = get_y_do_t_density_numeric_integration(
        y=y_sample,
        t=t_sample,
        y_mean_func=y_mean_func_gt,
        y_sd=y_sd_gt,
        z_mean=z_mean_gt,
        z_sd=z_sd_gt,
        n_z=n_z_gt,
    )

    assert densities_estimate.shape == densities_gt.shape == (t_sample.shape[0],), (
        densities_estimate.shape,
        densities_gt.shape,
        t_sample.shape[0],
    )

    samples = (y_max - y_min) * np.abs(densities_estimate - densities_gt)

    aid_estimate = np.mean(samples)
    aid_estimate_variance = np.var(samples, ddof=1) / samples.shape[0]

    return aid_estimate, aid_estimate_variance


def get_y_idrf_error_mlp(data: data_utils.MeDataset, model_mlp, data_generator):
    assert len(data.z.shape) == 2, data.z.shape
    assert len(data.t.shape) == 2, data.t.shape
    assert len(data.y.shape) == 2, data.y.shape

    y_mean_estimate = (
        model_mlp.forward(torch.from_numpy(data.z), torch.from_numpy(data.t))
        .detach()
        .numpy()
    )

    assert data.y.shape == y_mean_estimate.shape, data.y.shape + y_mean_estimate.shape

    y_mean_gt = data_generator.y_mean_func(data.z, data.t).reshape(-1, 1)
    assert data.y.shape == y_mean_gt.shape, data.y.shape + y_mean_gt.shape

    y_mean_error = np.sqrt(np.square(y_mean_gt - y_mean_estimate).mean())

    return y_mean_error


def get_y_sd_estimate_mlp(
    model: model_mlp.Mlp, test_data: data_utils.MeDataset, uses_t_instead_of_w: bool
):
    y_sd_estimate = model.get_y_sd_estimate(
        torch.from_numpy(test_data.z),
        torch.from_numpy(test_data.t if uses_t_instead_of_w else test_data.w),
        torch.from_numpy(test_data.y),
    )

    return y_sd_estimate


def y_idrf_uniform(
    z_for_param_ranges: np.ndarray,
    t_for_param_ranges: np.ndarray,
    model,
    model_type,
    y_gt_func,
    n_samples=10000,
):
    # Calculate the root mean squared error between the estimated and ground truth
    # Each sample is obtained by taking a uniformly random value for z (a vector) and t (a scalar)
    # so that every component of z is between its min and max value in z_for_param_ranges
    # and t is between its min and max value in t_for_param_ranges
    # z_for_param_ranges and t_for_param_ranges just contain some data, and this function finds the
    # minimums and maximums.
    # The ground truth is calculated using y_gt_func, which takes as arguments the z and t values
    # The estimated value is calculated using the model, which takes as arguments the z and t values

    z_min = np.min(z_for_param_ranges, axis=0)
    z_max = np.max(z_for_param_ranges, axis=0)
    t_min = np.min(t_for_param_ranges, axis=0)
    t_max = np.max(t_for_param_ranges, axis=0)

    z_samples = np.random.uniform(z_min, z_max, (n_samples, len(z_min))).astype(
        np.float32
    )
    t_samples = np.random.uniform(t_min, t_max, (n_samples, 1)).astype(np.float32)

    y_gt = y_gt_func(z_samples, t_samples)
    if model_type == "MLP":
        y_est = (
            model.forward(torch.from_numpy(z_samples), torch.from_numpy(t_samples))
            .detach()
            .numpy()
        )
    elif model_type == "CEME":
        y_est = (
            model.decoder.y_mean_estimate(
                torch.from_numpy(z_samples), torch.from_numpy(t_samples)
            )
            .detach()
            .numpy()
        )
    else:
        raise ValueError("Unknown model type: " + model_type)

    return np.sqrt(np.square(y_gt - y_est).mean())
