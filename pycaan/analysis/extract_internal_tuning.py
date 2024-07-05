# %% Imports
import yaml
import os
import numpy as np
from argparse import ArgumentParser
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.tuning import extract_tuning
from scipy.stats import zscore
import h5py


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--session_path", type=str, default="")
    args = parser.parse_args()
    return args


def extract_internal_tuning_session(data, params):
    if not os.path.exists(params["path_to_results"]):
        os.mkdir(params["path_to_results"])

    # Find folder
    working_directory = os.path.join(
        params["path_to_results"],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}",
    )

    if (
        not os.path.exists(os.path.join(working_directory, "internal_tuning.h5"))
        or params["overwrite_mode"] == "always"
    ) and (os.path.exists(os.path.join(working_directory, "embedding.h5"))):

        with h5py.File(
            os.path.join(working_directory, "embedding.h5"), "r"
        ) as embedding_file:

            # Filter epochs and normalize embedding
            train_embedding = embedding_file["train_embedding"][()]
            trainingFrames = embedding_file["trainingFrames"][()]
            std_embedding = zscore(train_embedding)

        bin_vec = (
            np.arange(
                -params["max_internal_distance"],
                params["max_internal_distance"] + params["internalBinSize"],
                params["internalBinSize"],
            ),
            np.arange(
                -params["max_internal_distance"],
                params["max_internal_distance"] + params["internalBinSize"],
                params["internalBinSize"],
            ),
            np.arange(
                -params["max_internal_distance"],
                params["max_internal_distance"] + params["internalBinSize"],
                params["internalBinSize"],
            ),
            np.arange(
                -params["max_internal_distance"],
                params["max_internal_distance"] + params["internalBinSize"],
                params["internalBinSize"],
            ),
        )

        (
            info,
            p_value,
            occupancy_frames,
            active_frames_in_bin,
            tuning_curves,
            peak_loc,
            peak_val,
        ) = extract_tuning(
            data["binaryData"][trainingFrames],
            std_embedding,
            np.ones(
                sum(trainingFrames), dtype="bool"
            ),  # Immobility already filtered
            bins=bin_vec,
        )

        with h5py.File(
            os.path.join(working_directory, "internal_tuning.h5"), "w"
        ) as f:
            f.create_dataset("info", data=info)
            f.create_dataset("p_value", data=p_value)
            f.create_dataset("occupancy_frames", data=occupancy_frames, dtype=int)
            f.create_dataset(
                "active_frames_in_bin", data=active_frames_in_bin, dtype=int
            )
            f.create_dataset("tuning_curves", data=tuning_curves)
            f.create_dataset("peak_loc", data=peak_loc)
            f.create_dataset("peak_val", data=peak_val)
            f.create_dataset("bins", data=bin_vec)

# If used as standalone script
if __name__ == "__main__":
    args = get_arguments()
    config = vars(args)

    with open("params.yaml", "r") as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    extract_internal_tuning_session(data, params)
