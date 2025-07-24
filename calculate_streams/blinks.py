# SPDX-FileCopyrightText: 2025 Tomi Božak, Jožef Stefan Institute
# SPDX-License-Identifier: MIT

from typing import Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import warnings

"""

## Blink Detection

Detect blinks in eye-tracking streams by finding periods where both gaze X and Y are missing (NaN) or where unusually large time-gaps occur between valid samples. 
Filter detected stretches to retain only those whose durations lie between configurable minimum and maximum thresholds.

### Algorithm

1. **Sort** the DataFrame by the specified `time_column`.  
2. **Time-based detection** (`detection="time"`):  
   - Drop rows where both X and Y are NaN.  
   - Compute the time difference Δ between each consecutive pair of valid samples.  
   - Convert Δ to the requested unit (seconds or milliseconds) if timestamps are datetime; otherwise assume numeric values are already in the correct unit.  
   - Flag any Δ such that `min_duration ≤ Δ ≤ max_duration` as a blink.  
3. **Frame-based detection** (`detection="frame"`):  
   - Build a boolean mask of rows where both X and Y are NaN.  
   - Locate start/end indices of each contiguous run of NaNs.  
   - Compute each run's length in frames and convert to time (using the sampling `frequency`) or leave as frame count (depending on `unit`).  
   - Keep runs whose durations fall within `[min_duration, max_duration]`.  
4. **Return** a list or DataFrame of blinks, including start/end times (or indices) and durations.

### Parameters

- `df` (`pd.DataFrame`): Eye-tracking data containing time and gaze columns.  
- `min_duration`, `max_duration` (`float`): Minimum and maximum blink duration thresholds, in the specified unit.  
- `unit` (`str`): `'frames'`, `'seconds'` (or synonyms `second`, `sec`), or `'milliseconds'` (`ms`).  
- `time_column` (`str`): Column name for timestamps (numeric or `datetime64[ns]`).  
- `gaze_x_column`, `gaze_y_column` (`str`): Column names for gaze X and Y coordinates.  
- `frequency` (`float`, optional): Sampling rate in Hz; required for frame-based detection when `unit != "frames"`.  
- `detection` (`"time"` or `"frame"`): Choose time-gap-based or NaN-run-based detection.  
- `return_indices` (`bool`): If `True`, include row indices of blink start/end.  
- `as_dataframe` (`bool`): If `True`, return results as a `pd.DataFrame` instead of a list of tuples.

### Usage Example

```python
from EyeTrackLib.calculate_streams.blinks import getBlinksVectorized
import pandas as pd

# Load eye-tracking CSV (timestamp parsed as datetime)
df = pd.read_csv("subject01_eyedata.csv", parse_dates=["timestamp"])

# Detect blinks lasting between 90 ms and 300 ms
blinks = getBlinksVectorized(
    df=df,
    min_duration=90,
    max_duration=300,
    unit="milliseconds",
    time_column="timestamp",
    gaze_x_column="gaze_x",
    gaze_y_column="gaze_y",
    detection="time",
    return_indices=False,
    as_dataframe=True
)

print(blinks)
#          start_time              end_time           duration
# 0  2025-07-02 12:00:01.230 2025-07-02 12:00:01.360   130.0
# 1  2025-07-02 12:00:05.480 2025-07-02 12:00:05.600   120.0

"""
def getBlinksVectorized(
    df: pd.DataFrame,
    min_duration: float,
    max_duration: float,
    unit: str,
    time_column: str,
    gaze_x_column: str,
    gaze_y_column: str,
    frequency: Optional[float] = None,
    return_indices: bool = False,
    as_dataframe: bool = False,
    detection: str = "time"
):
    """
    Detect blinks in eye-tracking streams using either time-gap or NaN-run detection.

    Algorithm:
      - Time-based mode ('time'): Drop rows where both gaze coords are NaN, compute Δt between consecutive valid samples, and flag any interval where min_duration ≤ Δt ≤ max_duration as a blink.
      - Frame-based mode ('frame'): Identify contiguous runs where both gaze coords are NaN, compute run lengths in frames or time (using frequency/unit), and filter runs between min_duration and max_duration.

    Parameters:
    - df (pd.DataFrame): Eyetracking data with time and gaze columns.
    - min_duration, max_duration (float): Duration thresholds in specified unit.
    - unit (str): One of 'frames', 'seconds', or 'milliseconds'. For time-based detection, `time_column` must be numeric and already expressed in this unit; durations are computed by simple subtraction.
    - frequency (float): Sampling rate in Hz. Only used for frame-based detection when unit!='frames'.
    - time_column (str): Column name for timestamps (in specified unit if unit!='frames' or in datetime format).
    - gaze_x_column, gaze_y_column (str): Column names for gaze coordinates.
    - return_indices (bool): If True, include start/end row indices in results.
    - as_dataframe (bool): If True, return results as a pandas DataFrame.
    - detection (str): 'frame' to use frame-based NaN-detection, 'time' to use time-interval detection (default).
    - Note: For 'time' detection, the function will drop NaN gaze rows, and compute `duration = t[i] - t[i-1]` directly, assuming your timestamps are in the correct unit.
    """
    # Validate presence of required columns, adjusting for detection mode
    if detection == "frame":
        required = {gaze_x_column, gaze_y_column}
    else:
        required = {time_column, gaze_x_column, gaze_y_column}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        warnings.warn("Input DataFrame is empty. No blinks detected.", UserWarning)
        if as_dataframe:
            cols = ['start_time','end_time','duration'] if not return_indices else ['start_idx','end_idx','start_time','end_time','duration']
            return pd.DataFrame(columns=cols)
        return []

    # Ensure sorted by time
    df = df.sort_values(time_column).reset_index(drop=True)

    # Blink detection based on chosen mode
    if detection not in ("time", "frame"):
        raise ValueError(f"Unsupported detection '{detection}'. Choose 'time' or 'frame'.")
    # Determine if time_column is numeric or datetime
    if is_numeric_dtype(df[time_column]):
        numeric_time = True
    elif is_datetime64_any_dtype(df[time_column]):
        numeric_time = False
    else:
        raise ValueError(
            f"time_column '{time_column}' must be numeric or datetime dtype; found {df[time_column].dtype}"
        )

    # Time-based detection: drop NaN rows and use time gaps
    if detection == "time":
        if unit == 'frames':
            raise ValueError("Unit 'frames' not supported for time-based detection.")
        clean_df = df.dropna(subset=[gaze_x_column, gaze_y_column], how="all")
        # If too few points, no blinks
        if len(clean_df) < 2:
            warnings.warn("Not enough valid data points for blink detection.", UserWarning)
            return pd.DataFrame(columns=(['start_time','end_time','duration'] 
                                         if as_dataframe and not return_indices else 
                                         ['start_idx','end_idx','start_time','end_time','duration'])) \
                   if as_dataframe else []
        # Keep original indices
        temp = clean_df.reset_index()
        results = []
        for i in range(1, len(temp)):
            prev_idx, prev_t = temp.at[i-1, 'index'], temp.at[i-1, time_column]
            curr_idx, curr_t = temp.at[i, 'index'], temp.at[i, time_column]
            # Compute delta
            delta = curr_t - prev_t
            if numeric_time:
                # Numeric time: delta already in the requested unit
                duration = delta
            else:
                # Datetime: convert timedelta to the requested unit
                if unit in ('seconds','second','sec'):
                    duration = delta.total_seconds()
                elif unit in ('milliseconds','ms'):
                    duration = delta.total_seconds() * 1000
                else:
                    raise ValueError(f"Unsupported unit '{unit}' for numeric time_column.")
            if min_duration <= duration <= max_duration:
                if return_indices:
                    results.append((prev_idx, curr_idx, prev_t, curr_t, duration))
                else:
                    results.append((prev_t, curr_t, duration))
        if as_dataframe:
            cols = ['start_time','end_time','duration'] if not return_indices else ['start_idx','end_idx','start_time','end_time','duration']
            return pd.DataFrame(results, columns=cols)
        return results

    # Frame-based detection: use NaN stretches
    elif detection == "frame":
        warnings.warn("Time column will not be used for frame-based detection; durations computed from frame counts.", UserWarning)
        invalid = df[gaze_x_column].isna() & df[gaze_y_column].isna()
        if frequency is None and unit != 'frames':
            raise ValueError("'frequency' must be provided for numeric time_column when unit is not 'frames'.")

        diff = invalid.astype(int).diff().fillna(0)
        starts = diff[diff == 1].index.tolist()
        ends = diff[diff == -1].index.tolist()
        if invalid.iloc[0]:
            starts.insert(0, 0)
        if invalid.iloc[-1]:
            ends.append(len(df) - 1)

        results = []

        for s, e in zip(starts, ends):
            # Compute duration from frame counts and frequency/unit
            frame_count = e - s + 1
            if unit == "frames":
                duration = frame_count
            else:
                if frequency is None:
                    raise ValueError("'frequency' must be provided for frame-based time units.")
                # Convert frame count to seconds
                dur_seconds = frame_count / frequency
                if unit in ("seconds","second","sec"):
                    duration = dur_seconds
                elif unit in ("milliseconds","ms","msec"):
                    duration = dur_seconds * 1000
                else:
                    raise ValueError(f"Unsupported unit '{unit}' for frame-based detection.")
            # Append result
            if return_indices:
                results.append((s, e, duration))
            else:
                if time_column in df.columns:
                    start_time = df.at[s, time_column]
                    end_time = df.at[e, time_column]
                    results.append((start_time, end_time, duration))
                else:
                    # If no time column, just append duration
                    results.append((duration,))

        if as_dataframe:
            cols = ['duration'] if not return_indices else ['start_idx','end_idx','duration']
            return pd.DataFrame(results, columns=cols)
        return results
