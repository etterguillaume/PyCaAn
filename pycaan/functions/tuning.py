import numpy as np

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr as corr

np.seterr(divide="ignore", invalid="ignore")  # Ignore zero divide warnings

def extract_tuning(binaryData, var, inclusion_ts, bins):
    _, numNeurons = binaryData.shape
    if var.ndim > 1:
        bin_dims = tuple([len(b) for b in bins])  # TODO assert that d_var == d_bins
        peak_loc = np.zeros((numNeurons, len(bin_dims)), dtype=int)
    else:
        bin_dims = len(bins)
        peak_loc = np.zeros((numNeurons, 1), dtype=int)
    binaryData = binaryData[inclusion_ts]
    var = var[inclusion_ts]
    
    active_frames_in_bin = np.zeros(
        (np.hstack((numNeurons, np.asarray(bin_dims) - 1))), dtype=int
    )

    info = np.zeros(numNeurons)
    p_value = np.zeros(numNeurons)
    peak_val = np.zeros(numNeurons)

    # Compute occupancy
    if var.ndim > 1:
        occupancy_frames = np.histogramdd(sample=var, bins=bins)[0]
    else:
        occupancy_frames = np.histogram(a=var, bins=bins)[0]

    # Digitize variable for info computations
    if var.ndim > 1:
        digitized = np.zeros(var.shape, dtype=int)
        for i in range(len(bins)):
            digitized[:, i] = np.digitize(var[:, i], bins[i], right=False)

        bin_vector = np.zeros(len(digitized))
        for i in range(len(digitized)):
            bin_vector[i] = np.ravel_multi_index(
                multi_index=digitized[i]-1, dims=bin_dims, mode='clip'
            )  # Convert to 1D

    else:
        bin_vector = np.digitize(var, bins, right=False)

    # Compute info and tuning curves for each neuron
    for neuron in range(numNeurons):
        if var.ndim > 1:
            active_frames_in_bin[neuron] = np.histogramdd(
                sample=var[binaryData[:, neuron]], bins=bins
            )[0]
        else:
            active_frames_in_bin[neuron] = np.histogram(
                a=var[binaryData[:, neuron]], bins=bins
            )[0]

        info[neuron] = adjusted_mutual_info_score(
            binaryData[:, neuron], bin_vector, average_method="min"
        )
        p_value[neuron] = chi2(binaryData[:, neuron][:, None], bin_vector[:, None])[1]
        peak_loc[neuron] = np.unravel_index(
            np.argmax(active_frames_in_bin[neuron], axis=None),
            active_frames_in_bin.shape[1::],
        )
        peak_val[neuron] = (
            active_frames_in_bin[(neuron,) + tuple(peak_loc[neuron])]
            / occupancy_frames[tuple(peak_loc[neuron])]
        )

    tuning_curves = (
        active_frames_in_bin / occupancy_frames
    )  # Likelihood = number of active frames in bin/occupancy

    return (
        info,
        p_value,
        occupancy_frames,
        active_frames_in_bin,
        tuning_curves,
        peak_loc,
        peak_val,
    )


def extract_discrete_tuning(binaryData, var, inclusion_ts):
    discrete_bin_vector = np.unique(var)
    binaryData = binaryData[inclusion_ts]
    var = var[inclusion_ts]
    numFrames, numNeurons = binaryData.shape
    active_frames_in_bin = np.zeros((numNeurons, len(discrete_bin_vector)), dtype=int)
    occupancy_frames = np.zeros(len(discrete_bin_vector), dtype=int)
    info = np.zeros(numNeurons)
    p_value = np.zeros(numNeurons)
    peak_val = np.zeros(numNeurons)
    peak_loc = np.zeros(numNeurons, dtype=int)

    # Compute occupancy
    occupancy_frames = np.bincount(var, minlength=len(discrete_bin_vector))

    # Bin activity
    for neuron in range(numNeurons):
        for x in discrete_bin_vector:
            frames_in_bin = np.where(var == x)[0]
            if frames_in_bin is not None:  # if bin has been explored
                active_frames_in_bin[neuron, x] = np.sum(
                    binaryData[frames_in_bin, neuron]
                )  # Total number of frames of activity in that bin

        info[neuron] = adjusted_mutual_info_score(
            binaryData[:, neuron], var, average_method="min"
        )
        p_value[neuron] = chi2(binaryData[:, neuron][:, None], var[:, None])[1]
        peak_loc[neuron] = np.argmax(active_frames_in_bin[neuron])
        peak_val[neuron] = (
            active_frames_in_bin[neuron, peak_loc[neuron]]
            / occupancy_frames[peak_loc[neuron]]
        )

    tuning_curves = (
        active_frames_in_bin / occupancy_frames
    )  # Likelihood = number of active frames in bin/occupancy
    return (
        info,
        p_value,
        occupancy_frames,
        active_frames_in_bin,
        tuning_curves,
        peak_loc,
        peak_val,
    )


def assess_covariate(
    var1, var2, inclusion_ts, var1_length, var1_bin_size, var2_length, var2_bin_size
):
    # Assess the amount of covariation between two variables (e.g. time and distance)
    var1_bin_vector = np.arange(0, var1_length + var1_bin_size, var1_bin_size)
    var2_bin_vector = np.arange(0, var2_length + var2_bin_size, var2_bin_size)
    var1 = var1[inclusion_ts]
    var2 = var2[inclusion_ts]
    digitized_var1 = np.zeros(
        len(var1), dtype=int
    )  # Vector that will specificy the bin# for each frame
    digitized_var2 = np.zeros(
        len(var2), dtype=int
    )  # Vector that will specificy the bin# for each frame

    # Digitize var1, var2 using params (length, bin_size)
    ct = 0
    if len(var1.shape) == 1:  # 1D variable #TODO fix 0D variables
        for i in range(len(var1_bin_vector) - 1):
            frames_in_bin = (var1 >= var1_bin_vector[i]) & (
                var1 < var1_bin_vector[i + 1]
            )
            digitized_var1[frames_in_bin] = ct
            ct += 1
    elif var1.shape[1] == 2:  # 2D variable
        for i in range(len(var1_bin_vector) - 1):
            for j in range(len(var1_bin_vector) - 1):
                frames_in_bin = (
                    (var1[:, 0] >= var1_bin_vector[i])
                    & (var1[:, 0] < var1_bin_vector[i + 1])
                    & (var1[:, 1] >= var1_bin_vector[j])
                    & (var1[:, 1] < var1_bin_vector[j + 1])
                )
                digitized_var1[frames_in_bin] = ct
                ct += 1

    ct = 0
    if len(var2.shape) == 1:  # 1D variable
        for i in range(len(var2_bin_vector) - 1):
            frames_in_bin = (var2 >= var2_bin_vector[i]) & (
                var2 < var2_bin_vector[i + 1]
            )
            digitized_var2[frames_in_bin] = ct
            ct += 1
    elif var2.shape[1] == 2:  # 2D variable
        for i in range(len(var2_bin_vector) - 1):
            for j in range(len(var2_bin_vector) - 1):
                frames_in_bin = (
                    (var2[:, 0] >= var2_bin_vector[i])
                    & (var2[:, 0] < var2_bin_vector[i + 1])
                    & (var2[:, 1] >= var2_bin_vector[j])
                    & (var2[:, 1] < var2_bin_vector[j + 1])
                )
                digitized_var1[frames_in_bin] = ct
                ct += 1

    # Compute AMI, p_value between two variables
    info = adjusted_mutual_info_score(
        digitized_var1, digitized_var2, average_method="min"
    )
    p_value = chi2(digitized_var1[:, None], digitized_var2[:, None])[1]
    correlation_results = corr(digitized_var1, digitized_var2)

    return info, p_value, correlation_results[0]


def extract_internal_info(
    data, params, inclusion_ts
):  # TODO alpha, very memory intensive currently
    retrospectivetime_bin_vector = np.arange(
        0,
        params["max_temporal_length"] + params["temporalBinSize"],
        params["temporalBinSize"],
    )
    prospectivetime_bin_vector = np.arange(
        0,
        params["max_temporal_length"] + params["temporalBinSize"],
        params["temporalBinSize"],
    )
    retrospectivedistance_bin_vector = np.arange(
        0,
        params["max_distance_length"] + params["spatialBinSize"],
        params["spatialBinSize"],
    )
    prospectivedistance_bin_vector = np.arange(
        0,
        params["max_distance_length"] + params["spatialBinSize"],
        params["spatialBinSize"],
    )
    speed_bin_vector = np.arange(
        0,
        params["max_velocity_length"] + params["velocityBinSize"],
        params["velocityBinSize"],
    )

    binaryData = data["binaryData"]

    var = np.vstack(
        (
            data["elapsed_time"],
            data["time2stop"],
            data["distance_travelled"],
            data["distance2stop"],
            data["velocity"],
        )
    ).T

    (
        internal_info,
        p_value,
        occupancy_frames,
        active_frames_in_bin,
        tuning_curve,
    ) = extract_tuning(
        binaryData,
        var,
        inclusion_ts,
        bins=(
            retrospectivetime_bin_vector,
            prospectivetime_bin_vector,
            retrospectivedistance_bin_vector,
            prospectivedistance_bin_vector,
            speed_bin_vector,
        ),
    )

    return internal_info, p_value


def extract_RWI(external_info, internal_info):
    # Internal info = place field info
    RWI = (external_info - internal_info) / (external_info + internal_info)

    return RWI
