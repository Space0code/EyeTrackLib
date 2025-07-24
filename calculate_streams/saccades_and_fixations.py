# SPDX-FileCopyrightText: 2025 Tomi Božak, Jožef Stefan Institute
# SPDX-License-Identifier: MIT

from typing import Dict, Tuple, Callable, Optional, Any
import numpy as np
import pandas as pd
import warnings
import logging
from logging import NullHandler
logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

######################################
########### MAIN FUNCTIONS ###########
######################################


def calculate_saccades_and_fixations(
    df: pd.DataFrame,
    gaze_x_column: str,
    gaze_y_column: str,
    time_column: str,
    window_start_time: float,
    window_end_time: float,
    algorithm: str = "I-DT",
    dispersion_threshold: float = 100.0,
    velocity_threshold: float = 30.0,
    min_fixation_duration: float = 100.0,
    min_saccade_duration: float = 0.0,
    max_saccade_duration: float = float("inf"),
    max_time_gap: float = 400.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate saccades and fixations from Tobii eye tracker data based on the provided `algorithm`.

    Warning:
      - Assumes time is in milliseconds!
      - Assumes that invalid data is replaced with np.nan or float("nan") values before calling this function.

    Supported algorithms:
      - Dispersion-Threshold Identification (I-DT) (default)
      - Velocity Threshold Identification (I-VT) (Note: I-VT is in development and may not work as expected. Use I-DT for best results.)

    Parameters:
      df (pd.DataFrame): DataFrame containing eye tracker data.
      gaze_x_column (str): Name of the gaze X coordinate column.
      gaze_y_column (str): Name of the gaze Y coordinate column.
      time_column (str): Name of the time column.
      window_start_time (float): Start time of the window of interest.
      window_end_time (float): End time of the window of interest.
      algorithm (str): Algorithm to use for detecting saccades and fixations. Supported values are 'I-VT' and 'I-DT'.
      dispersion_threshold (float): Threshold for detecting fixations (I-DT). Ignored if algorithm is 'I-VT'.
      velocity_threshold (float): Threshold for detecting saccades (I-VT). Ignored if algorithm is 'I-DT'.
      min_fixation_duration (float): Minimum duration for a fixation in milliseconds.
      max_time_gap (float): Maximum time gap between two consecutive points in milliseconds.

    Returns:
      tuple:
        - saccades (pd.DataFrame): DataFrame of detected saccades with columns ['Saccade_StartTime','Saccade_EndTime','Path']
        - fixations (pd.DataFrame): DataFrame of detected fixations with columns ['Fixation_StartTime', 'Fixation_EndTime', 'Fixation_CentroidX', 'Fixation_StdX', 'Fixation_CentroidY', 'Fixation_StdY']
    """

    if algorithm == "I-DT":
        saccades, fixations = i_dt(
            df,
            gaze_x_column=gaze_x_column,
            gaze_y_column=gaze_y_column,
            time_column=time_column,
            dispersion_threshold=dispersion_threshold,
            max_time_gap=max_time_gap,
            min_fixation_duration=min_fixation_duration,
            min_saccade_duration=min_saccade_duration,
            max_saccade_duration=max_saccade_duration,
            window_start_time=window_start_time,
            window_end_time=window_end_time,
            dispersion=default_dispersion,
        )
    elif algorithm == "I-VT":
        saccades, fixations = i_vt(
            df,
            window_start_time,
            window_end_time,
            max_time_gap,
            velocity_threshold,
            min_fixation_duration,
            gaze_x_column,
            gaze_y_column,
            time_column,
        )
    else:
        raise ValueError(
            "Unsupported algorithm. Supported values are 'I-VT' and 'I-DT'."
        )

    return saccades, fixations


def i_vt(
    df: pd.DataFrame,
    window_start_time: float,
    window_end_time: float,
    max_time_gap: float,
    velocity_threshold: float,
    min_fixation_duration: float,
    gaze_x_column: str,
    gaze_y_column: str,
    time_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    I-VT (Velocity Threshold Identification) algorithm for detecting saccades and fixations from eye tracker data.

    Parameters:
      df (pd.DataFrame): DataFrame containing eye tracker data.
      window_start_time (float): Start time of the window of interest.
      window_end_time (float): End time of the window of interest.
      max_time_gap (float): Maximum time gap between consecutive points in ms.
      velocity_threshold (float): Threshold velocity for saccade detection (pixels/ms).
      min_fixation_duration (float): Minimum duration for a fixation in ms.
      gaze_x_column (str): Name of the gaze X coordinate column.
      gaze_y_column (str): Name of the gaze Y coordinate column.
      time_column (str): Name of the time column.

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: saccades_df, fixations_df
        - saccades_df: DataFrame of detected saccades with columns ['Saccade_StartTime','Saccade_EndTime','Path']
        - fixations_df: DataFrame of detected fixations with columns ['Fixation_StartTime', 'Fixation_EndTime', 'Fixation_CentroidX', 'Fixation_StdX', 'Fixation_CentroidY', 'Fixation_StdY']
    """

    # 0. Filter the data based on the window start and end times and drop nan rows because we cannot calculate velocity for them
    df = df.loc[df[time_column].between(window_start_time, window_end_time)].copy()
    df.dropna(subset=[gaze_x_column, gaze_y_column], inplace=True)

    # 1. Calculate time differences and velocities between consecutive points - both are one datapoint shorter than other data!
    df["time_diff"] = df[time_column].diff()

    # TODO - check if this func works properly
    df["velocity"] = np.append(
        [0], getVelocities(df[gaze_x_column], df[gaze_y_column], df[time_column])
    )

    # 2. Detect saccades and fixations based on the velocity threshold (I-VT algorithm)
    saccades = []
    fixations = []
    fixation_start = None  # None means there is no fixation currently
    cur_saccade = []
    prev_i = None  # index of the previous row because we dropped some rows inbetween due to missing data
    preceded_by_fixation = (
        False  # Flag to check if the current saccade is preceded by a fixation
    )

    for i, row in df.iterrows():
        if prev_i is not None:
            fixation_end = df.loc[prev_i][time_column]  # possible fixation-end time

        # Check if the time gap is too big - should we end the current fixation/saccade or drop it?
        if row["time_diff"] > max_time_gap:
            # End current fixation or saccade before the big gap
            if fixation_start is not None:
                # End of (possible) fixation
                preceded_by_fixation = _endFixation(
                    fixations,
                    fixation_start,
                    fixation_end,
                    df,
                    prev_i,
                    min_fixation_duration,
                    gaze_x_column,
                    gaze_y_column,
                    time_column,
                )
                fixation_start = None
            else:
                # End current saccade
                fixation_start = _endSaccade(
                    row,
                    cur_saccade,
                    saccades,
                    preceded_by_fixation,
                    gaze_x_column,
                    gaze_y_column,
                    time_column,
                )
                cur_saccade = []
                preceded_by_fixation = False

        # Check if the velocity is greater than the threshold
        elif row["velocity"] > velocity_threshold:
            if fixation_start is not None:
                preceded_by_fixation = _endFixation(
                    fixations,
                    fixation_start,
                    fixation_end,
                    df,
                    prev_i,
                    min_fixation_duration,
                    gaze_x_column,
                    gaze_y_column,
                    time_column,
                )

                fixation_start = None

            # Add the current point to the current saccade if a fixation already happened before else ignore
            if preceded_by_fixation:
                if cur_saccade == []:
                    # Add fixation point to the saccade TODO: should we add the last fixation point to the saccade?
                    first_row = df.loc[prev_i]
                    cur_saccade.append(
                        (
                            first_row[time_column],
                            first_row[gaze_x_column],
                            first_row[gaze_y_column],
                            # first_row["velocity"], - TODO - think which rows' velocities to add!
                        )
                    )

                # Building the saccade path
                cur_saccade.append(
                    (
                        row[time_column],
                        row[gaze_x_column],
                        row[gaze_y_column],
                        # row["velocity"],
                    )
                )

        # Else, we are in a (middle of or at the start of a) fixation
        else:
            if fixation_start is None:
                fixation_start = _endSaccade(
                    row, cur_saccade, saccades, preceded_by_fixation, gaze_x_column, gaze_y_column, time_column
                )
                cur_saccade = []  # Reset the saccade list
            else:
                preceded_by_fixation = True  # Continue the current fixation

        prev_i = i

    # DO NOT end the last fixation/saccade because the data might be incomplete

    # Convert fixations list to DataFrame
    fixations_df = pd.DataFrame(fixations, columns=[
        "Fixation_StartTime", "Fixation_EndTime",
        "Fixation_CentroidX", "Fixation_StdX",
        "Fixation_CentroidY", "Fixation_StdY"
    ])

    # Convert saccades list to DataFrame
    saccades_df = pd.DataFrame(saccades, columns=[
        "Saccade_StartTime", "Saccade_EndTime", "Path"
    ])

    return saccades_df, fixations_df


def default_dispersion(max_x: float, min_x: float, max_y: float, min_y: float) -> float:
    """Default dispersion function: (max_x-min_x)+(max_y-min_y)."""
    return (max_x - min_x) + (max_y - min_y)

def i_dt(
    df: pd.DataFrame,
    gaze_x_column: str,
    gaze_y_column: str,
    time_column: str,
    dispersion_threshold: float,
    max_time_gap: float = float("inf"),
    min_fixation_duration: float = 0.0,
    min_saccade_duration: float = 0.0,
    max_saccade_duration: float = float("inf"),
    window_start_time: Optional[float] = None,
    window_end_time: Optional[float] = None,
    dispersion: Callable[[float, float, float, float], float] = default_dispersion,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    I-DT (Dispersion Threshold Identification) algorithm for detecting fixations and saccades from eye tracker data.

    Parameters:
      df (pd.DataFrame): DataFrame containing eye tracker data.
      gaze_x_column (str): Name of the gaze X coordinate column.
      gaze_y_column (str): Name of the gaze Y coordinate column.
      time_column (str): Name of the time column.
      dispersion_threshold (float): Dispersion threshold for fixation detection.
      max_time_gap (float): Maximum time gap between consecutive points in ms.
      min_fixation_duration (float): Minimum duration for a fixation in ms.
      min_saccade_duration (float): Minimum duration for a saccade in ms.
      max_saccade_duration (float): Maximum duration for a saccade in ms.
      window_start_time (Optional[float]): Start time of the window of interest.
      window_end_time (Optional[float]): End time of the window of interest.
      dispersion (Callable): Function to compute dispersion.

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: saccades_df, fixations_df
        - saccades_df: DataFrame of detected saccades with columns ['Saccade_StartTime','Saccade_EndTime','Path']
        - fixations_df: DataFrame of detected fixations (all features as columns)
    """

    df = _init_df_check_fixations(df, gaze_x_column, gaze_y_column, time_column, dispersion_threshold, max_time_gap, min_fixation_duration, window_start_time, window_end_time, dispersion)

    # Dict with features of fixations to return
    fixations: Dict[str, Any] = {}

    # Points of a potential fixation
    still_points: Dict[str, list] = _reset_still_points(time_column, gaze_x_column, gaze_y_column)

    # Iterate over the rows of the DataFrame
    N = df.shape[0]
    row_indx = 0
    while row_indx < N:
        row = df.iloc[row_indx]

        if _potential_fixation_exists(still_points, time_column):
            # Happens when the previous still_points were not enough to declare a fixation, so the first point was popped in the previous iteration
            max_x = max(still_points[gaze_x_column])
            min_x = min(still_points[gaze_x_column])
            max_y = max(still_points[gaze_y_column])
            min_y = min(still_points[gaze_y_column])
            disp = dispersion(max_x, min_x, max_y, min_y)
        else:
            # initialize variables
            disp = 0  # dispersion
            max_x = None
            max_y = None
            min_x = None
            min_y = None

        # Add still (fixation) points to still_points until dispersion_threshold is reached
        while row_indx < N and disp < dispersion_threshold:
            row = df.iloc[row_indx]  # update row
            t = row[time_column]
            if (
                _potential_fixation_exists(still_points, time_column)
                and (t - still_points[time_column][-1]) > max_time_gap
            ):
                # if the time gap between the previous and current datapoint is too big, end the current (potential) fixation at the previous point
                break

            x = row[gaze_x_column]
            y = row[gaze_y_column]

            max_x = max(max_x, x) if max_x is not None else x
            min_x = min(min_x, x) if min_x is not None else x
            max_y = max(max_y, y) if max_y is not None else y
            min_y = min(min_y, y) if min_y is not None else y

            # calculate dispersion: (max_x - min_x) + (max_y - min_y)
            disp = dispersion(max_x, min_x, max_y, min_y)

            # if dispersion thresh is not reached, add the point to still_points (fixation)
            if disp < dispersion_threshold:
                still_points[time_column].append(t)
                still_points[gaze_x_column].append(x)
                still_points[gaze_y_column].append(y)
                row_indx += 1

        # Dispersion threshold or max_time_gap was reached.
        if row_indx < N and _potential_fixation_exists(still_points, time_column):
            fix_start = still_points[time_column][0]
            fix_end = still_points[time_column][-1]
            dt = fix_end - fix_start  # delta time

            # if the window is long enough, declare a fixation
            if dt >= min_fixation_duration and dt > 0:
                # Declare a fixation
                fixations = _update_fixations(fixations, still_points, gaze_x_column, gaze_y_column, time_column, dispersion=dispersion)

                # Reset still_points
                still_points = _reset_still_points(time_column, gaze_x_column, gaze_y_column)
            else:
                # Window is not long enough to declare a fixation 
                
                # If the time gap between the previous and current datapoint is too big and the fixation is too short, we drop (skip) the observation --> reset still_points
                if (row[time_column] - still_points[time_column][-1]) > max_time_gap:
                    still_points = _reset_still_points(time_column, gaze_x_column, gaze_y_column)

                # If the time gap is not too big, remove the first point and continue, because only the dispersion threshold was reached
                else:
                    for key in still_points:
                        logger.debug(
                            f"Removing first point from {key} in still_points because the potential fixation is too short."
                        )
                        still_points[key].pop(0)


    # END THE LAST FIXATION, ALTHOUGH IT MAY BE INCOMPLETE
    if _potential_fixation_exists(still_points, time_column):
        fix_start = np.min(still_points[time_column])
        fix_end = np.max(still_points[time_column])
        dt = fix_end - fix_start  # delta time

        # if the window is long enough, declare a fixation
        if dt >= min_fixation_duration and dt > 0:
            # Declare a fixation
            fixations = _update_fixations(fixations, still_points, gaze_x_column, gaze_y_column, time_column, dispersion=dispersion)

            # Reset still_points
            still_points = _reset_still_points(time_column, gaze_x_column, gaze_y_column)


    # Compute raw saccades list
    saccades_list = calculate_saccades_from_idt_fixations(
        df,
        fixations,
        window_start_time,
        window_end_time,
        max_saccade_duration,
        gaze_x_column,
        gaze_y_column,
        time_column,
        min_saccade_duration=min_saccade_duration,
    )

    # Convert fixations dict to DataFrame
    fixations_df = pd.DataFrame(fixations)

    # Convert saccades list to DataFrame
    saccades_df = pd.DataFrame([
        {
            "Saccade_StartTime": start,
            "Saccade_EndTime": end,
            "Path": path,
        }
        for start, end, path in saccades_list
    ])

    return saccades_df, fixations_df



def calculate_saccades_from_idt_fixations(
    df: pd.DataFrame,
    idt_fixations: Dict[str, Any],
    window_start_time: float,
    window_end_time: float,
    max_saccade_duration: float,
    gaze_x_column: str,
    gaze_y_column: str,
    time_column: str,
    min_saccade_duration: float = 0.0,
) -> list[tuple[float, float, list[tuple[float, float, float]]]]:
    """
    Calculate saccades from the fixations detected by the I-DT algorithm.

    Parameters:
      df (pd.DataFrame): DataFrame containing eye tracker data.
      idt_fixations (Dict[str, Any]): Dictionary with features of fixations (from i_dt).
      window_start_time (float): Start time of the window of interest.
      window_end_time (float): End time of the window of interest.
      max_saccade_duration (float): Maximum duration for a saccade in ms.
      gaze_x_column (str): Name of the gaze X coordinate column.
      gaze_y_column (str): Name of the gaze Y coordinate column.
      time_column (str): Name of the time column.
      min_saccade_duration (float): Minimum duration for a saccade in ms.

    Returns:
      List[Tuple[float, float, List[Tuple[float, float, float]]]]:
        List of saccades, each as (start_time, end_time, path), where path is a list of (time, x, y) tuples.
    """

    df = _init_df_check_saccades(df, time_column, gaze_x_column, gaze_y_column, window_start_time, window_end_time, idt_fixations, min_saccade_duration, max_saccade_duration)
    
    saccades_list = [] # format: [(start_time, end_time, path=[(time, x, y), ...])]
    # Pre-extract arrays to speed up indexing
    times = df[time_column].values
    xs    = df[gaze_x_column].values
    ys    = df[gaze_y_column].values
    
    for i in range(len(idt_fixations["Fixation_StartTime"]) - 1):
        start = idt_fixations["Fixation_EndTime"][i]
        end = idt_fixations["Fixation_StartTime"][i + 1]

        if start < window_start_time:
            continue  # skip if we are not yet in the window of interest

        if end > window_end_time:
            break

        # boolean mask for the region
        mask = (times >= start) & (times < end)
        if not mask.any():
            continue  # no samples in this region

        if min_saccade_duration < end - start < max_saccade_duration:
            # build the path as a list of (time, x, y)
            path = list(zip(times[mask], xs[mask], ys[mask]))
            saccades_list.append((start, end, path))

    return saccades_list



######################################
# Helper functions
######################################


def _endFixation(
    fixations: list,
    fixation_start: float,
    fixation_end: float,
    df: pd.DataFrame,
    prev_i: int,
    min_fixation_duration: float,
    gaze_x_column: str,
    gaze_y_column: str,
    time_column: str,
) -> bool:
    """
    End the current fixation and append it to the list of fixations `fixations` if it meets the minimum duration requirement.

    Parameters:
      fixations (list): List to store the detected fixations.
      fixation_start (float): Timestamp when the current fixation started.
      fixation_end (float): Timestamp when the current fixation ended.
      df (pd.DataFrame): DataFrame containing eye tracker data.
      prev_i (int): Index of the previous row in the DataFrame.
      min_fixation_duration (float): Minimum duration for a fixation in milliseconds.
      gaze_x_column (str): Name of the gaze X coordinate column.
      gaze_y_column (str): Name of the gaze Y coordinate column.
      time_column (str): Name of the time column.

    Returns:
      bool: True if a fixation was appended, False otherwise.
    """

    # End of the current fixation

    duration = fixation_end - fixation_start
    if duration >= min_fixation_duration:
        fixation_x = df.loc[
            (fixation_start <= df[time_column]) & (df[time_column] <= fixation_end),
            gaze_x_column,
        ]
        fixation_centroid_x = np.nanmean(fixation_x)
        fixation_x_std = np.nanstd(fixation_x)

        fixation_y = df.loc[
            (fixation_start <= df[time_column]) & (df[time_column] <= fixation_end),
            gaze_y_column,
        ]
        fixation_centroid_y = fixation_y.mean()
        fixation_y_std = fixation_y.std()

        fixation = (
            fixation_start,
            fixation_end,
            fixation_centroid_x,
            fixation_x_std,
            fixation_centroid_y,
            fixation_y_std,
        )

        fixations.append(fixation)

        return True
    return False


def _endSaccade(
    row: pd.Series,
    cur_saccade: list[tuple[float, float, float]],
    saccades: list,
    preceded_by_fixation: bool,
    gaze_x_column: str,
    gaze_y_column: str,
    time_column: str,
) -> float:
    """
    End the current saccade (if there is one) and append it to the list of saccades `saccades`.

    Parameters:
      row (pd.Series): Current row of the DataFrame.
      cur_saccade (list): List of points that belong to the current saccade.
      saccades (list): List to store the detected saccades.
      preceded_by_fixation (bool): True if this saccade is preceded by a fixation.
      gaze_x_column (str): Name of the gaze X coordinate column.
      gaze_y_column (str): Name of the gaze Y coordinate column.
      time_column (str): Name of the time column.

    Returns:
      float: The new fixation_start timestamp.
    """
    fixation_start = row[time_column]
    if len(cur_saccade) > 0 and preceded_by_fixation:
        sac_start_time = cur_saccade[0][0]
        sac_end_time = fixation_start
        sac_path = cur_saccade + [(sac_end_time, row[gaze_x_column], row[gaze_y_column])]
        saccades.append((sac_start_time, sac_end_time, sac_path))
    return fixation_start


def getDistances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances between consecutive x, y points.
    Parameters:
      x (np.ndarray): Array of x coordinates.
      y (np.ndarray): Array of y coordinates.
    Returns:
      np.ndarray: Array of distances.
    """
    return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

def getVelocities(x: np.ndarray, y: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Compute velocities between consecutive gaze points.
    Parameters:
      x (np.ndarray): Array of x coordinates.
      y (np.ndarray): Array of y coordinates.
      times (np.ndarray): Array of timestamps.
    Returns:
      np.ndarray: Array of velocities (length one less than input arrays).
    """
    return getDistances(x, y) / np.diff(times)


def _init_df_check_general(
    df: pd.DataFrame,
    time_column: str,
    gaze_x_column: str,
    gaze_y_column: str,
    window_start_time: Optional[float],
    window_end_time: Optional[float],
) -> pd.DataFrame:
    """
    General DataFrame check for the required columns and emptiness.
    Parameters:
      df (pd.DataFrame): DataFrame to check.
      time_column (str): Name of the time column.
      gaze_x_column (str): Name of the gaze X column.
      gaze_y_column (str): Name of the gaze Y column.
      window_start_time (Optional[float]): Start time of window.
      window_end_time (Optional[float]): End time of window.
    Returns:
      pd.DataFrame: Checked and filtered DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty. Please provide a DataFrame with data.")

    if not isinstance(time_column, str):
        raise ValueError("parameter time_column must be a string representing the column name.")

    if time_column not in df.columns:
        raise ValueError(f"DataFrame is missing required column: {time_column}")

    if not (df[time_column].dtype.kind in ("i", "f", "M")):
        raise TypeError("time_column column must be numeric or datetime")
    
    # sort by time_column to ensure the data is in chronological order
    df = df.sort_values(by=time_column, inplace=False)
    if not isinstance(gaze_x_column, str) or not isinstance(gaze_y_column, str):
        raise ValueError("gaze_x_column, gaze_y_column, and time_column must be strings representing column names.")

    required = {gaze_x_column, gaze_y_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    
        
    # If no window is specified, use the whole DataFrame
    if window_start_time is None:
        window_start_time = df[time_column].min()
        warnings.warn(
            f"No window_start_time specified. Using the minimum time in the DataFrame: {window_start_time}",
            stacklevel=2
        )
    if window_end_time is None:
        window_end_time = df[time_column].max()
        warnings.warn(
            f"No window_end_time specified. Using the maximum time in the DataFrame: {window_end_time}",
            stacklevel=2
        )

    if window_start_time >= window_end_time:
        raise ValueError("window_start_time must be less than window_end_time.")

    # Get only the data in the window of interest
    df = (
        df.loc[
            (df[time_column] >= window_start_time)
            & (df[time_column] <= window_end_time)
        ]
        .copy()
    )

    # drop rows with missing data
    df.dropna(subset=[gaze_x_column, gaze_y_column], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def _init_df_check_fixations(
    df: pd.DataFrame,
    gaze_x_column: str,
    gaze_y_column: str,
    time_column: str,
    dispersion_threshold: float,
    max_time_gap: float,
    min_fixation_duration: float,
    window_start_time: Optional[float],
    window_end_time: Optional[float],
    dispersion: Callable[[float, float, float, float], float],
) -> pd.DataFrame:
    """
    DataFrame check for the required columns and emptiness for fixation calculation.
    Parameters:
      df (pd.DataFrame): DataFrame to check.
      gaze_x_column (str): Name of the gaze X column.
      gaze_y_column (str): Name of the gaze Y column.
      time_column (str): Name of the time column.
      dispersion_threshold (float): Dispersion threshold.
      max_time_gap (float): Maximum time gap.
      min_fixation_duration (float): Minimum fixation duration.
      window_start_time (Optional[float]): Start time of window.
      window_end_time (Optional[float]): End time of window.
      dispersion (Callable): Dispersion function.
    Returns:
      pd.DataFrame: Checked and filtered DataFrame.
    """
    
    df = _init_df_check_general(df, time_column, gaze_x_column, gaze_y_column, window_start_time, window_end_time)


    for name, val in [
        ("dispersion_threshold", dispersion_threshold),
        ("min_fixation_duration",    min_fixation_duration),
        ("max_time_gap",             max_time_gap)
    ]:
        if val < 0:
            raise ValueError(f"{name!r} must be >= 0, got {val}")
        
    if dispersion_threshold == 0:
        warnings.warn(
            "dispersion_threshold is set to 0, which means that most likely no fixations will be detected. ",
            stacklevel=2
        )
    if max_time_gap == 0:
        warnings.warn(
            "max_time_gap is set to 0, which means that most likely no fixations will be detected. ",
            stacklevel=2
        )

    # check if the dispersion function is callable
    if not callable(dispersion):
        raise TypeError("dispersion must be a callable function that takes four arguments: max_x, min_x, max_y, min_y.")

    return df

def _init_df_check_saccades(
    df: pd.DataFrame,
    time_column: str,
    gaze_x_column: str,
    gaze_y_column: str,
    window_start_time: float,
    window_end_time: float,
    idt_fixations: Dict[str, Any],
    min_saccade_duration: float,
    max_saccade_duration: float,
) -> pd.DataFrame:
    """
    DataFrame check for the required columns and emptiness for saccades calculation.
    Parameters:
      df (pd.DataFrame): DataFrame to check.
      time_column (str): Name of the time column.
      gaze_x_column (str): Name of the gaze X column.
      gaze_y_column (str): Name of the gaze Y column.
      window_start_time (float): Start time of window.
      window_end_time (float): End time of window.
      idt_fixations (Dict[str, Any]): Fixations dictionary.
      min_saccade_duration (float): Minimum saccade duration.
      max_saccade_duration (float): Maximum saccade duration.
    Returns:
      pd.DataFrame: Checked and filtered DataFrame.
    """
    df = _init_df_check_general(df, time_column, gaze_x_column, gaze_y_column, window_start_time, window_end_time)

    for name, val in [
        ("min_saccade_duration", min_saccade_duration),
        ("max_saccade_duration", max_saccade_duration)
    ]:
        if val < 0:
            raise ValueError(f"{name!r} must be >= 0, got {val}")
        
    if min_saccade_duration >= max_saccade_duration:
        raise ValueError(
            f"min_saccade_duration ({min_saccade_duration}) must be less than max_saccade_duration ({max_saccade_duration})."
        )
    if max_saccade_duration == 0:
        warnings.warn(
            "max_saccade_duration is set to 0, which means that most likely no saccades will be detected.",
            stacklevel=2
        )

    if idt_fixations is None or not isinstance(idt_fixations, dict):
        raise ValueError("idt_fixations must be a dictionary returned from i_dt function.")
    if "Fixation_StartTime" not in idt_fixations:
        raise ValueError("idt_fixations must contain 'Fixation_StartTime' key.")
    if "Fixation_EndTime" not in idt_fixations:
        raise ValueError("idt_fixations must contain 'Fixation_EndTime' key.")

    return df

def _update_fixations(
    fixations: Dict[str, Any],
    still_points: Dict[str, list],
    gaze_x_column: str,
    gaze_y_column: str,
    time_column: str,
    dispersion: Callable[[float, float, float, float], float] = default_dispersion,
) -> Dict[str, Any]:
    features = [
        "Fixation_StartTime",
        "Fixation_EndTime",
        "Fixation_Duration",
        "Fixation_CentroidX",
        "Fixation_CentroidY",
        "Fixation_StdX",
        "Fixation_StdY",
        "Fixation_Dispersion",
        "Fixation_RangeX",
        "Fixation_RangeY",
        "Fixation_MaxX",
        "Fixation_MinX",
        "Fixation_MaxY",
        "Fixation_MinY",
        "Fixation_FirstX",
        "Fixation_FirstY",
        "Fixation_LastX",
        "Fixation_LastY",
    ]

    # add the features to the fixations dictionary
    for feature in features:
        fixations.setdefault(feature, [])

    fix_start = np.min(still_points[time_column])
    fix_end = np.max(still_points[time_column])
    fixations["Fixation_StartTime"].append(fix_start)
    fixations["Fixation_EndTime"].append(fix_end)
    fixations["Fixation_Duration"].append(fix_end - fix_start)
    fixations["Fixation_CentroidX"].append(np.mean(still_points[gaze_x_column]))
    fixations["Fixation_CentroidY"].append(np.mean(still_points[gaze_y_column]))
    fixations["Fixation_StdX"].append(np.std(still_points[gaze_x_column]))
    fixations["Fixation_StdY"].append(np.std(still_points[gaze_y_column]))

    max_x = max(still_points[gaze_x_column])
    min_x = min(still_points[gaze_x_column])
    max_y = max(still_points[gaze_y_column])
    min_y = min(still_points[gaze_y_column])
    fixations["Fixation_Dispersion"].append(dispersion(max_x, min_x, max_y, min_y))
    fixations["Fixation_RangeX"].append(max_x - min_x)
    fixations["Fixation_RangeY"].append(max_y - min_y)
    fixations["Fixation_MaxX"].append(max_x)
    fixations["Fixation_MinX"].append(min_x)
    fixations["Fixation_MaxY"].append(max_y)
    fixations["Fixation_MinY"].append(min_y)

    # save the first and last points of the fixation
    first_x = still_points[gaze_x_column][0]
    first_y = still_points[gaze_y_column][0]
    last_x = still_points[gaze_x_column][-1]
    last_y = still_points[gaze_y_column][-1]
    fixations["Fixation_FirstX"].append(first_x)
    fixations["Fixation_FirstY"].append(first_y)
    fixations["Fixation_LastX"].append(last_x)
    fixations["Fixation_LastY"].append(last_y)

    return fixations

def _potential_fixation_exists(still_points: Dict[str, list], time_column: str) -> bool:
    """
    Check if there are any points in still_points (a potential fixation cluster).
    Parameters:
      still_points (Dict[str, list]): Still points dictionary.
      time_column (str): Name of the time column.
    Returns:
      bool: True if still_points exist, False otherwise.
    """
    return still_points[time_column] != []

def _reset_still_points(time_column: str, gaze_x_column: str, gaze_y_column: str) -> Dict[str, list]:
    """
    Reset the still_points dictionary to empty lists.
    Parameters:
      time_column (str): Name of the time column.
      gaze_x_column (str): Name of the gaze X column.
      gaze_y_column (str): Name of the gaze Y column.
    Returns:
      Dict[str, list]: Dictionary of empty lists for still points.
    """
    return {time_column: [], gaze_x_column: [], gaze_y_column: []}
