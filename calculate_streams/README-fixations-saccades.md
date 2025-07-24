<!-- SPDX-FileCopyrightText: 2025 Tomi Božak, Jožef Stefan Institute -->
<!-- SPDX-License-Identifier: MIT -->

# Fixations & Saccades

This module provides **fixation** and **saccade** detection from raw gaze `(x, y)` and timestamps.  
It implements:

- **I-DT (Dispersion-Threshold Identification)** – **primary & recommended**.
- **I-VT (Velocity-Threshold Identification)** – **beta**, provided for experimentation.

All timestamps are **assumed to be in milliseconds** and gaze coordinates in **pixels** (unless you pre-convert).

---

## Algorithm Overview

**I-DT (Dispersion-Threshold Identification)**: The algorithm iteratively expands a time window over consecutive gaze points, calculating the spatial dispersion (sum of the range in X and Y coordinates). When this dispersion remains below the specified threshold for at least the minimum fixation duration, those points are classified as a fixation; saccades are (roughly) defined as the intervals between successive fixations.

**I-VT (Velocity-Threshold Identification)**: This algorithm computes the instantaneous velocity between successive gaze samples and labels segments with velocities below the threshold and lasting at least the minimum fixation duration as fixations, while segments where velocity exceeds the threshold are marked as saccades. Warning: I-VT is in beta mode, we do not recommend it!

---
## Notes, Limitations & Warnings

- **Unit assumptions**: Time must be in milliseconds and gaze coordinates in pixels. Convert datetimes or visual degrees beforehand.
- **I-VT is beta**: Use I-DT for production.
- **Saccades (I-DT)** only detected between consecutive fixations; no leading/trailing saccades are captured. (This is a feature, not a bug.)
- **Zero thresholds**: Setting `dispersion_threshold = 0` or `max_time_gap = 0` will prevent any fixations from being detected (warnings are issued).
- **NaN handling**: Both algorithms drop rows with `NaN` gaze values; detect blinks separately if needed.
- **Performance**: Loops are adequate for moderate datasets; consider vectorized or JIT approaches for high-frequency, long-duration recordings.

## TL;DR

```python
from EyeTrackLib.calculate_streams.saccades_and_fixations import calculate_saccades_and_fixations

saccades_df, fixations_df = calculate_saccades_and_fixations(
    df=df,
    gaze_x_column="gaze_x",
    gaze_y_column="gaze_y",
    time_column="time_ms",
    window_start_time=0.0,
    window_end_time=df["time_ms"].max(),
    algorithm="I-DT",             # <— recommended (I-VT is beta)
    dispersion_threshold=100.0,    # px
    min_fixation_duration=100.0,   # ms
    max_time_gap=400.0             # ms
)

