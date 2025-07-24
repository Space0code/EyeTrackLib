<!-- SPDX-FileCopyrightText: 2025 Tomi Božak, Jožef Stefan Institute -->
<!-- SPDX-License-Identifier: MIT -->
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
