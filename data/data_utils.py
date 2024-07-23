from typing import Optional

import numpy as np
import pandas as pd
import torch

import utils


def numpy_to_me_dataset(arr):
    assert arr.shape[1] == 4 and arr.shape[2] == 1, arr.shape
    return MeDataset(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


class MeDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self,
        z: np.ndarray,
        t: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ):
        n_data = z.shape[0]
        utils.check_shape(z, (n_data, -1))
        utils.check_shape(t, (n_data, 1))
        utils.check_shape(w, (n_data, 1))
        utils.check_shape(y, (n_data, 1))
        self.z, self.t, self.w, self.y = z, t, w, y

        assert z.dtype == np.float32, z.dtype
        assert t.dtype == np.float32, t.dtype
        assert w.dtype == np.float32, w.dtype
        assert y.dtype == np.float32, y.dtype

        if weights is None:
            self.weights = np.ones((len(self), 1), dtype=np.float32)
        else:
            self.weights = weights
        self.weights = (
            self.weights * len(self) / np.sum(self.weights)
        )  # weights sum to len(self)

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.z[idx]),
            torch.tensor(self.t[idx]),
            torch.tensor(self.w[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.weights[idx]),
        )

    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.concatenate((self.z, self.t, self.w, self.y, self.weights), axis=1),
            columns=["z", "t", "w", "y", "weights"],
        )

    def sorted_by_t(self):
        ind = np.argsort(self.t, axis=0).flatten()
        return MeDataset(
            z=self.z[ind],
            t=self.t[ind],
            w=self.w[ind],
            y=self.y[ind],
            weights=self.weights[ind],
        )

    def sorted_by_z(self):
        ind = np.argsort(self.z, axis=0).flatten()
        return MeDataset(
            z=self.z[ind],
            t=self.t[ind],
            w=self.w[ind],
            y=self.y[ind],
            weights=self.weights[ind],
        )
