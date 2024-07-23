import numpy as np
import torch
import pytest

import utils


def test_check_shape_with_correct_numpy_array():
    arr = np.array([[1, 2], [3, 4]])
    shape = (2, 2)
    # Should pass without assertion error
    utils.check_shape(arr, shape)


def test_check_shape_with_incorrect_numpy_array():
    arr = np.array([1, 2, 3, 4])
    shape = (2, 2)
    with pytest.raises(AssertionError):
        utils.check_shape(arr, shape)


def test_check_shape_with_correct_torch_tensor():
    arr = torch.tensor([[1, 2], [3, 4]])
    shape = (2, 2)
    # Should pass without assertion error
    utils.check_shape(arr, shape)


def test_check_shape_with_incorrect_torch_tensor():
    arr = torch.tensor([1, 2, 3, 4])
    shape = (2, 2)
    with pytest.raises(AssertionError):
        utils.check_shape(arr, shape)


def test_check_shape_with_wildcard():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    shape = (2, -1)
    # Should pass without assertion error
    utils.check_shape(arr, shape)


def test_check_shape_with_list_in_shape():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    shape = (2, [1, 3])
    # Should pass without assertion error
    utils.check_shape(arr, shape)


def test_check_shape_with_incorrect_list_in_shape():
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    shape = (2, [1, 3])
    with pytest.raises(AssertionError):
        utils.check_shape(arr, shape)
