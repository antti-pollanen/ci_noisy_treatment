import os
from typing import Literal, Optional

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing  # "import sklearn" does not work for some reason

import data.data_utils as data_utils


class Generator:
    def __init__(self, outcome) -> None:
        data_dir = os.path.dirname(os.path.abspath(__file__)) + "/education_wage_data/"
        df: pd.DataFrame = pd.read_csv(data_dir + "/education_wage_data.csv")
        df: pd.DataFrame = self._preprocess_df(df, outcome)

        treatment: Literal["educ"] = "educ"
        t = df[treatment].to_numpy().astype("float32").reshape(-1, 1)
        y = df[outcome].to_numpy().astype("float32").reshape(-1, 1)
        z = df.drop(columns=[treatment, outcome]).to_numpy().astype("float32")

        scaler = preprocessing.StandardScaler()
        self.z = scaler.fit_transform(z)
        self.t = scaler.fit_transform(t)
        self.w = np.zeros_like(self.t)  # not used
        self.y = scaler.fit_transform(y)

        self.first_new_data_index: int = 0
        self.n_total = df.shape[0]

    def _preprocess_df(self, df: pd.DataFrame, outcome: str) -> pd.DataFrame:
        wage_to_not_use: Literal["wage", "lwage"] = (
            "wage" if outcome == "lwage" else "lwage"
        )
        features_to_drop: list[str] = [
            "fatheduc",  # correlation with education, lots of nan values
            "motheduc",  # correlation with education, lots of nan values
            "weight",  # this is NLS sampling weight, so not to be used as a feature. Also not to be used as weights in training as I'd prefer to give equal weight to different types of people.
            "KWW",
            "IQ",
            # "libcrd14" # high correlation, but causally justified to condition for this to remove spurious correlations
            "exper",
            "expersq",
            "exp_bin",
            wage_to_not_use,
        ]

        df = df.drop(columns=features_to_drop)
        df = df.dropna()

        df = df.sample(frac=1).reset_index(drop=True)  # shuffle

        return df

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
        assert False, "not implemented"
        return 0.0

    def get_y_sd(self):
        assert False, "not implemented"
        return 0.0


def get_generator(**kwargs) -> Generator:
    return Generator(**kwargs)
