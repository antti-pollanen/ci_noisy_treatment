import contextlib
import json
import random
from typing import Tuple, Union, List

import numpy as np
import torch


def check_shape(
    arr: np.ndarray | torch.Tensor, shape: Tuple[Union[int, List[int]], ...]
) -> None:
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().detach().numpy()
    assert arr.ndim == len(shape), f"Got shape {arr.shape}, expected {shape}"
    for i, expected_dim in enumerate(shape):
        if expected_dim == -1:
            continue

        if isinstance(expected_dim, list):
            assert (
                arr.shape[i] in expected_dim
            ), f"Got shape {arr.shape}, expected {shape}"
        else:
            assert (
                arr.shape[i] == expected_dim
            ), f"Got shape {arr.shape}, expected {shape}"


def safe_softplus(x: float, limit: int = 30) -> np.ndarray:
    return np.where(x > limit, x, np.log1p(np.exp(x)))


def set_random_seeds(val: int = 0) -> None:
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)


@contextlib.contextmanager
def temp_numpy_seed(seed: int) -> None:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def get_json_serializable_object(obj):
    if type(obj) is dict:
        return {key: get_json_serializable_object(val) for key, val in obj.items()}
    elif type(obj) in (list, tuple):
        return [get_json_serializable_object(obj_i) for obj_i in obj]
    elif np.issubdtype(type(obj), np.integer):
        return int(obj)
    elif np.issubdtype(type(obj), np.floating):
        return float(obj)
    elif not is_jsonable(obj):
        return str(obj)
    else:
        return obj
