import numpy as np

import data.data_utils as data_utils
import utils


def create_se_kernel(X1, X2, alpha=1, scale=1, eps=1e-7):
    """returns the NxM squared exponential kernel matrix between the two sets of input X1 and X2

    arguments:
    X1    -- N length array or NxD matrix
    X2    -- M length array or MxD matrix
    alpha -- scalar
    scale -- scalar
    eps -- added to the diagonal of the kernel matrix to ensure it is positive definite

    returns NxM matrix
    """

    # use float64 since with float32 we need to use a larger eps to ensure positive definiteness
    # which causes "noisy" output functions as correlations between points very close to each
    # other are not very close to one anymore
    # assert X1.dtype == np.float64, X1.dtype
    # assert X2.dtype == np.float64, X2.dtype

    X1 = X1.view()
    X2 = X2.view()

    if len(X1.shape) == 1:
        X1.shape = (-1, 1)

    if len(X2.shape) == 1:
        X2.shape = (-1, 1)

    assert X1.shape[1] == X2.shape[1]

    print("Creating kernel of size", X1.shape[0], "x", X2.shape[0], "x", X1.shape[1])

    differences = X1[:, np.newaxis, :] - X2
    norms = np.linalg.norm(differences, axis=2)

    ret = alpha * np.exp(-np.square(norms) / (2 * scale * scale))

    if X1.shape == X2.shape:
        ret += np.identity(X1.shape[0]) * eps

    print("Size of kernel created in memory:", ret.size * ret.itemsize)

    return ret


class GP_T_MeanFunc:
    def __init__(
        self, z_grid, kernel_grid_inv, t_means_grid, kernel_alpha, kernel_scale
    ):
        self.z_grid = z_grid
        self.t_kernel_grid_inv = kernel_grid_inv
        self.t_means_grid = t_means_grid

        self.kernel_alpha = kernel_alpha
        self.kernel_scale = kernel_scale

    def __call__(self, z):
        # formula for posterior mean from eq. 28 in https://mycourses.aalto.fi/pluginfile.php/1688729/mod_resource/content/1/lec1.pdf:
        # conditional indepence or not is irrelevant here since the calculation is deterministic
        kernel_actual_grid = create_se_kernel(
            z, self.z_grid, self.kernel_alpha, self.kernel_scale
        )
        t_cov_coefficient = kernel_actual_grid @ self.t_kernel_grid_inv
        return t_cov_coefficient @ self.t_means_grid


class GP_T_SD_Func:
    def __init__(
        self, z_grid, kernel_grid_inv, t_sds_grid, t_sd_mean, kernel_alpha, kernel_scale
    ):
        self.z_grid = z_grid
        self.t_kernel_grid_inv = kernel_grid_inv
        self.t_sds_grid = t_sds_grid
        self.t_sd_mean = t_sd_mean

        self.kernel_alpha = kernel_alpha
        self.kernel_scale = kernel_scale

    def __call__(self, z):
        # formula for posterior mean from eq. 28 in https://mycourses.aalto.fi/pluginfile.php/1688729/mod_resource/content/1/lec1.pdf:
        # conditional indepence or not is irrelevant here since the calculation is deterministic
        kernel_actual_grid = create_se_kernel(
            z, self.z_grid, self.kernel_alpha, self.kernel_scale
        )
        t_cov_coefficient = kernel_actual_grid @ self.t_kernel_grid_inv
        return utils.safe_softplus(
            t_cov_coefficient @ (self.t_sds_grid - self.t_sd_mean) + self.t_sd_mean
        )


class GP_Y_MeanFunc:
    def __init__(
        self,
        zt_grid,
        y_kernel_grid_inv,
        y_means_grid,
        kernel_alpha,
        kernel_scale,
        max_array_size=10**8,  # 10**9,
    ):
        self.zt_grid = zt_grid
        self.y_kernel_grid_inv = y_kernel_grid_inv
        self.y_means_grid = y_means_grid

        self.kernel_alpha = kernel_alpha
        self.kernel_scale = kernel_scale

        self.max_array_size = max_array_size

    def __call__(self, z: float | np.ndarray, t: np.ndarray):
        # t: a numpy array of any shape
        # z: scalar or numpy array of same shape as t
        # formula for posterior mean from eq. 28 in https://mycourses.aalto.fi/pluginfile.php/1688729/mod_resource/content/1/lec1.pdf:
        # conditional indepence or not is irrelevant here since the calculation is deterministic

        input_shape = t.shape
        if np.isscalar(z):
            z = np.zeros(shape=t.shape) + z

        assert t.shape == z.shape

        z = z.view()
        t = t.view()

        z.shape = -1
        t.shape = -1

        assert len(t.shape) == 1

        chunk_size = self.max_array_size // (self.zt_grid.shape[0] * 2)

        # kernel_actual_grid = create_se_kernel(
        #     np.stack((z, t), axis=1), self.zt_grid, self.kernel_alpha, self.kernel_scale
        # )
        # return (
        #     kernel_actual_grid @ self.y_kernel_grid_inv @ self.y_means_grid
        # ).reshape(input_shape)

        # perform the above calculation that is in comments in chunks
        # to avoid memory errors
        print("GP_Y_MeanFunc.__call__: allocating array for results", flush=True)

        y_means = np.zeros(shape=t.shape)
        for i in range(0, t.shape[0], chunk_size):
            print(
                "Calculating y mean in chunks for chunk number ",
                i // chunk_size,
                " / ",
                t.shape[0] // chunk_size,
            )
            kernel_actual_grid = create_se_kernel(
                np.stack(
                    (z[i : i + chunk_size], t[i : i + chunk_size]), axis=1
                ),  # creates a copy, but hard to avoid
                self.zt_grid,
                self.kernel_alpha,
                self.kernel_scale,
            )
            y_means[i : i + chunk_size] = (
                kernel_actual_grid @ self.y_kernel_grid_inv @ self.y_means_grid
            )

        return y_means.reshape(input_shape)


class GpDataGenerator:
    def __init__(
        self,
        n_total,
        z_mean,
        z_sd,
        relative_w_sd,
        relative_y_sd,
        kernel_alpha=1,
        kernel_scale=1,
        n_grid_points=1000,
        local_seed=0,
    ):
        assert (
            n_total >= 100
        ), "This class is for generating properly sized datasets only"
        print("GpDataGenerator local seed:", local_seed)
        self.n_total = n_total

        self.z_mean = z_mean
        self.z_sd = z_sd

        self.n_grid_points = n_grid_points

        self.z_grid = None
        self.t_kernel_grid_inv = None

        self.t_means_grid = None
        self.t_sds_grid = None

        self.t_sd_mean = 1

        self.zt_grid = None
        self.y_kernel_grid_inv = None

        self.relative_w_sd = relative_w_sd
        self.relative_y_sd = relative_y_sd

        self.absolute_w_sd = None
        self.absolute_y_sd = None

        self.all_z = None
        self.all_t = None
        self.all_w = None
        self.all_y = None

        self.first_new_data_index = 0

        self.kernel_alpha = kernel_alpha
        self.kernel_scale = kernel_scale

        self.local_seed = local_seed

        self.t_mean_func = None
        self.t_sd_func = None

        self.y_mean_func = None

        self.data_pregenerated = False

    def get_w_sd(self):
        return self.absolute_w_sd

    def get_y_sd(self):
        return self.absolute_y_sd

    def _pregenerate_data(self):
        # generate z
        # find the correct grid for t_mean_func GP based on min and max z
        # then do multivariate Gaussian sampling on that grid to get the grid points (need mean and covariance matrix), separate GP for both mean and noise
        # then get the actual t samples as posterior mean + posterior noise, should approximate the actual GP well enough

        # generate w samples, by finding the relative mean

        # do same thing for y as for t, but now the input is 2-dimensional
        with utils.temp_numpy_seed(self.local_seed):
            self.all_z = np.random.normal(
                loc=self.z_mean, scale=self.z_sd, size=self.n_total
            )

            z_grid_min = self.all_z.min() - 0.25 * (self.all_z.max() - self.all_z.min())
            z_grid_max = self.all_z.max() + 0.25 * (self.all_z.max() - self.all_z.min())

            self.z_grid = np.linspace(
                start=z_grid_min,
                stop=z_grid_max,
                num=self.n_grid_points,
            )

            t_kernel_grid = create_se_kernel(self.z_grid, self.z_grid)

            t_mean_means = np.zeros(shape=self.n_grid_points)
            t_sd_means = np.zeros(shape=self.n_grid_points) + self.t_sd_mean

            self.t_means_grid = np.random.multivariate_normal(
                mean=t_mean_means, cov=t_kernel_grid
            )
            self.t_sds_grid = np.log1p(
                np.exp(
                    np.random.multivariate_normal(mean=t_sd_means, cov=t_kernel_grid)
                )
            )

            self.t_kernel_grid_inv = np.linalg.inv(t_kernel_grid)

            self.t_mean_func = GP_T_MeanFunc(
                z_grid=self.z_grid,
                kernel_grid_inv=self.t_kernel_grid_inv,
                t_means_grid=self.t_means_grid,
                kernel_alpha=self.kernel_alpha,
                kernel_scale=self.kernel_scale,
            )
            self.t_sd_func = GP_T_SD_Func(
                z_grid=self.z_grid,
                kernel_grid_inv=self.t_kernel_grid_inv,
                t_sds_grid=self.t_sds_grid,
                t_sd_mean=self.t_sd_mean,
                kernel_alpha=self.kernel_alpha,
                kernel_scale=self.kernel_scale,
            )

            self.all_t = np.random.normal(
                loc=self.t_mean_func(self.all_z),
                scale=self.t_sd_func(self.all_z),
                size=self.n_total,
            )

            w_mean_sd = np.std(self.all_t, ddof=1)
            self.absolute_w_sd = w_mean_sd * self.relative_w_sd

            self.all_w = np.random.normal(
                loc=self.all_t, scale=self.absolute_w_sd, size=self.n_total
            )

            t_grid_min = self.all_t.min() - 0.25 * (self.all_t.max() - self.all_t.min())
            t_grid_max = self.all_t.max() + 0.25 * (self.all_t.max() - self.all_t.min())

            z_scale = np.linspace(
                start=z_grid_min,
                stop=z_grid_max,
                num=int(np.floor(np.sqrt(self.n_grid_points))),
            )
            t_scale = np.linspace(
                start=t_grid_min,
                stop=t_grid_max,
                num=int(np.floor(np.sqrt(self.n_grid_points))),
            )

            zz, tt = np.meshgrid(z_scale, t_scale)

            self.zt_grid = np.stack((zz, tt), axis=2).reshape((-1, 2))

            y_kernel_grid = create_se_kernel(self.zt_grid, self.zt_grid)

            y_mean_means = np.zeros(shape=y_kernel_grid.shape[0])

            self.y_means_grid = np.random.multivariate_normal(
                mean=y_mean_means, cov=y_kernel_grid
            )

            self.y_kernel_grid_inv = np.linalg.inv(y_kernel_grid)

            self.y_mean_func = GP_Y_MeanFunc(
                zt_grid=self.zt_grid,
                y_kernel_grid_inv=self.y_kernel_grid_inv,
                y_means_grid=self.y_means_grid,
                kernel_alpha=self.kernel_alpha,
                kernel_scale=self.kernel_scale,
            )
            y_means = self.y_mean_func(self.all_z, self.all_t)

            y_mean_sd = np.std(y_means, ddof=1)
            self.absolute_y_sd = y_mean_sd * self.relative_y_sd

            self.all_y = np.random.normal(
                loc=y_means, scale=self.absolute_y_sd, size=self.n_total
            )

        self.data_pregenerated = True

    def generate_data(self, n, z_const=None, take_from_start=False):
        if not self.data_pregenerated:
            self._pregenerate_data()

        if not z_const is None:
            return self.generate_additional_data_with_constant_z(n, z_const)

        if take_from_start:
            start_i = 0
            end_i = n
        else:
            assert (
                n + self.first_new_data_index <= self.n_total
            ), "attempted to generate more data than what is pregenerated"
            start_i = self.first_new_data_index
            self.first_new_data_index += n
            end_i = self.first_new_data_index
        return data_utils.MeDataset(
            self.all_z[start_i:end_i].reshape(-1, 1).astype(np.float32),
            self.all_t[start_i:end_i].reshape(-1, 1).astype(np.float32),
            self.all_w[start_i:end_i].reshape(-1, 1).astype(np.float32),
            self.all_y[start_i:end_i].reshape(-1, 1).astype(np.float32),
        )
