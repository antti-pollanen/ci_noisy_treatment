import os
from typing import Optional

import numpy as np
import torch

import data.data_utils as data_utils


class Generator:
    def __init__(self, relative_w_sd) -> None:
        data_dir = os.path.dirname(os.path.abspath(__file__)) + "/education_wage_data/"
        dataset: data_utils.MeDataset = torch.load(
            data_dir
            + f"education_wage_data_augmented_noise_{int(100*relative_w_sd)}_percent.pickle"
        )
        print("Loaded the dataset with noise {}%".format(int(100 * relative_w_sd)))

        with open(data_dir + "y_sd.txt", "r", encoding="utf-8") as f:
            self.y_sd_gt = float(f.read())

        self.relative_w_sd = relative_w_sd
        self.y_gt_model = torch.load(
            data_dir + "education_wage_augmenting_model.pickle"
        )

        self.z = dataset.z
        self.t = dataset.t
        self.w = dataset.w
        self.y = dataset.y

        assert self.z.dtype == np.float32
        assert self.t.dtype == np.float32
        assert self.w.dtype == np.float32
        assert self.y.dtype == np.float32

        self.first_new_data_index = 0
        self.n_total = self.z.shape[0]

        # no shuffling as we assume the data is already shuffled
        # also important this way to retain the same (train,validate)-test split

    def generate_data(
        self, n: int, z_const: Optional[int] = None, take_from_start: bool = False
    ) -> data_utils.MeDataset:
        assert z_const is None, "generating data with constant z not supported"

        if take_from_start:
            assert n <= self.n_total
            start_i = 0
            end_i = n
        else:
            assert n + self.first_new_data_index <= self.n_total, (
                "attempted to generate data over limit, "
                + str(n)
                + " + "
                + str(self.first_new_data_index)
                + " > "
                + str(self.n_total)
            )
            start_i = self.first_new_data_index
            self.first_new_data_index += n
            end_i = self.first_new_data_index

        return data_utils.MeDataset(
            self.z[start_i:end_i],
            self.t[start_i:end_i].reshape(-1, 1),
            self.w[start_i:end_i].reshape(-1, 1),
            self.y[start_i:end_i].reshape(-1, 1),
        )

    def get_w_sd(self):
        return self.relative_w_sd * np.std(self.t)

    def get_y_sd(self):
        return self.y_sd_gt

    def y_mean_func(self, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.y_gt_model(torch.tensor(z), torch.tensor(t)).numpy()
