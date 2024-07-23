from typing import Any

from sklearn.model_selection import ParameterGrid


def combine_param_dicts(dict1, dict2):
    new = dict1.copy()
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(value, dict):
                if not isinstance(dict1[key], dict):
                    raise ValueError(f"Key {key} is already in dict1 and is not a dict")
                new[key] = combine_param_dicts(dict1[key], value)
            else:
                raise ValueError(f"Key {key} is already in dict1 and is not a dict")
        else:
            new[key] = value

    return new


def get_run_params_by_index(constant_params, param_grid, index) -> dict[str, Any]:
    grid = ParameterGrid(param_grid)
    # grid[index] consists of a list of dictionaries, combine these with constant_params
    params = constant_params.copy()
    params["param_grid"] = grid[index % len(grid)]
    for axis_name, axis_values in grid[index % len(grid)].items():
        axis_values = axis_values.copy()
        axis_values.pop("name")
        params = combine_param_dicts(params, axis_values)
    return params


def get_num_distinct_runs(param_grid) -> int:
    grid = ParameterGrid(param_grid)
    return len(grid)
