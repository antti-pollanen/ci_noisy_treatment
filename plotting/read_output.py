import json
from typing import Any

import pandas as pd


def load_common_output(test_paths: list[str]) -> tuple[list[str], list[int]]:
    test_names = []
    max_task_ids = []

    for test_path in test_paths:
        common_output_filename = test_path + "/common_output.json"

        with open(common_output_filename, "r", encoding="utf-8") as read_file:
            common_output = json.load(read_file)

        test_names.append(common_output["experiment_name"])
        max_task_ids.append(common_output["max_task_id"])

    return test_names, max_task_ids


def read_data_as_dicts(
    file_name_common_parts: list[str], max_task_ids: list[int]
) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []

    for file_name_common_part, max_task_id in zip(file_name_common_parts, max_task_ids):
        print(
            "Reading data for", file_name_common_part, "with max task id", max_task_id
        )
        case_id = 0
        failed_to_open_ids = []
        while case_id <= max_task_id:
            if case_id % 1000 == 0:
                print("Reading case", case_id)
            read_filename = "{}_{}.json".format(file_name_common_part, case_id)
            try:
                with open(read_filename, "r") as read_file:
                    data.append(json.load(read_file))
            except IOError:
                failed_to_open_ids.append(case_id)
            case_id += 1

        print(
            "For",
            file_name_common_part,
            "failed to open json file for ids",
            failed_to_open_ids,
        )
    return data


def flatten_dict(d: dict) -> dict:
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(
                [
                    (k + "." + inner_k, inner_v)
                    for inner_k, inner_v in flatten_dict(v).items()
                ]
            )
        else:
            items.append((k, v))
    return dict(items)


def get_data_as_df(data_as_dicts: list[dict]) -> pd.DataFrame:
    data_as_dicts = [flatten_dict(data_dict) for data_dict in data_as_dicts]
    return pd.DataFrame(data_as_dicts)


def read_data_as_df(test_paths: list[str]) -> pd.DataFrame:
    _, max_task_ids = load_common_output(test_paths)
    data_as_dicts = read_data_as_dicts(
        [test_path + "/out_case" for test_path in test_paths], max_task_ids
    )
    return get_data_as_df(data_as_dicts)
