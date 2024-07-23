import argparse
import json
import os.path
import subprocess


def read_and_validate_previous_common_output(experiment_name, common_output_filename):
    try:
        with open(common_output_filename, "r", encoding="utf-8") as read_file:
            data = json.load(read_file)
    except IOError as e:
        raise FileNotFoundError(
            f"Unable to open for reading the file {common_output_filename}: {e}"
        ) from e

    if experiment_name != data.get("experiment_name"):
        raise ValueError(
            f"Name of experiment: {experiment_name}, experiment name on file: {data.get('experiment_name')}"
        )

    if "max_task_id" not in data:
        raise ValueError(f"max_task_id is missing in {common_output_filename}")

    new_start_id = data["max_task_id"] + 1
    return new_start_id


def write_new_common_output(experiment_name, max_task_id, common_output_filename):
    data = {"experiment_name": experiment_name, "max_task_id": max_task_id}

    try:
        with open(common_output_filename, "w", encoding="utf-8") as write_file:
            json.dump(data, write_file)
    except IOError as e:
        raise IOError(
            f"Unable to open for writing the file {common_output_filename}: {e}"
        ) from e


def ceildiv(a, b):
    return -(a // -b)


def slurm_batch_run(
    experiment_name,
    out_subdir_name,
    num_array_jobs_requested,
    additional_experiment_arguments,
):
    """
    Submits batch jobs to Slurm based on the experiment configuration.

    This function configures and submits Slurm batch jobs for different experiments
    by setting memory requirements, time requirements, and other parameters based on
    the experiment name. It also handles the creation of necessary output directories
    and the preparation of common output files (output that is shared between the
    array jobs).

    Parameters:
    - experiment_name (str): The name of the experiment to run.
    - out_subdir_name (str): The name of the output subdirectory to store results.
    - num_array_jobs_requested (int): The number of array jobs requested to run.
    - additional_experiment_arguments (list): Additional arguments to pass to the experiment script.

    Raises:
    - ValueError: If an unknown experiment name is provided.
    - AssertionError: If the number of tasks to run at once is not divisible by the chunk size,
                      or if the start ID is not divisible by the number of tasks to run at once.
    """

    # The variable num_tasks_to_run_at_once is the number of tasks with distinct grid parameters,
    # the idea being that we want to run an equal number of tasks for each parameter combination.
    if experiment_name == "synthetic":
        mem_required = "20G"
        time_required_per_array_job = "00:30:00"
        time_required_per_chunk = "00:30:00"
        experiment_python_script = "slurm/grid_experiment.py"
        num_tasks_to_run_at_once = 216
        chunk_size = 1
    elif experiment_name == "education_wage":
        mem_required = "4G"
        time_required_per_array_job = "00:30:00"
        time_required_per_chunk = "01:30:00"
        experiment_python_script = "slurm/grid_experiment.py"
        num_tasks_to_run_at_once = 24  # run 24*200 = 4800 in total (200 of these are dummy runs for CEME+ with no measurement error)
        chunk_size = 4
    else:
        raise ValueError(f"Unknown experiment_name: {experiment_name}")

    print("Running", experiment_name)

    output_dir = "output/" + out_subdir_name + "/"

    common_output_filename = output_dir + "common_output.json"

    if os.path.isfile(common_output_filename):
        start_id = read_and_validate_previous_common_output(
            experiment_name, common_output_filename
        )
    else:
        start_id = 0

    assert num_tasks_to_run_at_once % chunk_size == 0

    num_array_jobs_actual = (
        ceildiv(num_array_jobs_requested, num_tasks_to_run_at_once)
        * num_tasks_to_run_at_once
    )

    num_chunks = num_array_jobs_actual // chunk_size
    max_task_id = start_id + num_array_jobs_actual - 1

    assert start_id % num_tasks_to_run_at_once == 0
    start_chunk_id = start_id // chunk_size

    subprocess.run(["mkdir", "-p", output_dir], check=True)

    write_new_common_output(experiment_name, max_task_id, common_output_filename)

    subprocess.run(
        [
            "sbatch",
            "--array",
            str(start_chunk_id) + "-" + str(start_chunk_id + num_chunks - 1),
            "--mem=" + mem_required,
            "--time=" + time_required_per_chunk,
            "--output=" + output_dir + "sbatch_%a.out",
            "slurm/array_job.sh",
            experiment_name,
            experiment_python_script,
            str(chunk_size),
            time_required_per_array_job,
            output_dir,
            " ".join([str(x) for x in additional_experiment_arguments]),
        ],
        check=True,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a batch of Slurm jobs for a given experiment."
    )
    parser.add_argument("experiment_name", type=str, help="The name of the experiment.")
    parser.add_argument(
        "out_subdir_name", type=str, help="The name of the output subdirectory."
    )
    parser.add_argument(
        "num_array_jobs_requested", type=int, help="The number of array jobs requested."
    )
    parser.add_argument(
        "additional_experiment_arguments",
        nargs="*",
        help="Additional arguments for the experiment script.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    slurm_batch_run(
        args.experiment_name,
        args.out_subdir_name,
        args.num_array_jobs_requested,
        args.additional_experiment_arguments,
    )
