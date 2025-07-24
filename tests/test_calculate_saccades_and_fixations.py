# SPDX-FileCopyrightText: 2025 Tomi Božak, Jožef Stefan Institute
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
import pytest
from calculate_streams.saccades_and_fixations import calculate_saccades_and_fixations  # Replace with actual module name

# Helper function to generate random gaze data
def generate_random_gaze(n_points, n_fixations, max_saccade_duration):
    times = np.cumsum(np.random.uniform(10, 100, n_points))  # Cumulative time in ms
    gaze_x = []
    gaze_y = []
    current_x = np.random.uniform(0, 1920)
    current_y = np.random.uniform(0, 1080)
    for _ in range(n_fixations):
        # Fixation: small variations
        fixation_duration = np.random.uniform(100, 500)
        n_fix_points = max(1, int(fixation_duration / (times[1] - times[0]) if len(times) > 1 else 1))
        for _ in range(n_fix_points):
            gaze_x.append(current_x + np.random.normal(0, 5))
            gaze_y.append(current_y + np.random.normal(0, 5))
        # Saccade: large jump
        saccade_duration = np.random.uniform(20, max_saccade_duration)
        n_sacc_points = max(1, int(saccade_duration / (times[1] - times[0]) if len(times) > 1 else 1))
        for _ in range(n_sacc_points):
            current_x += np.random.uniform(-100, 100)
            current_y += np.random.uniform(-100, 100)
            gaze_x.append(current_x)
            gaze_y.append(current_y)
    # Adjust to match n_points
    if len(gaze_x) > n_points:
        gaze_x = gaze_x[:n_points]
        gaze_y = gaze_y[:n_points]
    elif len(gaze_x) < n_points:
        gaze_x.extend([gaze_x[-1]] * (n_points - len(gaze_x)))
        gaze_y.extend([gaze_y[-1]] * (n_points - len(gaze_y)))
    df = pd.DataFrame({'t': times[:n_points], 'x': gaze_x, 'y': gaze_y})
    return df

# Random tests (80 tests)
def test_random_scenarios():
    for _ in range(80):
        n_points = np.random.randint(100, 1000)
        n_fixations = np.random.randint(1, 10)
        df = generate_random_gaze(n_points, n_fixations, max_saccade_duration=100)
        algorithm = np.random.choice(['I-VT', 'I-DT'])
        if algorithm == 'I-VT':
            velocity_threshold = np.random.uniform(10, 50)
            saccades, fixations = calculate_saccades_and_fixations(
                df, 'x', 'y', 't', df['t'].min(), df['t'].max(), algorithm=algorithm,
                velocity_threshold=velocity_threshold, min_fixation_duration=50.0
            )
        else:
            dispersion_threshold = np.random.uniform(50, 200)
            saccades, fixations = calculate_saccades_and_fixations(
                df, 'x', 'y', 't', df['t'].min(), df['t'].max(), algorithm=algorithm,
                dispersion_threshold=dispersion_threshold, min_fixation_duration=50.0
            )
        # General assertions
        assert not saccades.empty or not fixations.empty, "At least one event should be detected"
        if not saccades.empty:
            assert all(saccades['Saccade_StartTime'] < saccades['Saccade_EndTime']), "Saccade start times must precede end times"
        if not fixations.empty:
            assert all(fixations['Fixation_StartTime'] < fixations['Fixation_EndTime']), "Fixation start times must precede end times"

# Hard tests (20 edge cases)
def test_constant_gaze():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=0.0
    )
    assert len(fixations) == 1, "Should detect one fixation"
    assert fixations.iloc[0]['Fixation_StartTime'] == times[0], "Fixation should start at the beginning"
    assert fixations.iloc[0]['Fixation_EndTime'] == times[-1], "Fixation should end at the last timestamp"
    assert saccades.empty, "No saccades should be detected"

def test_single_saccade():
    times = np.arange(0, 100, 10)
    xs = [100.0]*5 + [200.0]*5
    ys = [200.0]*5 + [300.0]*5
    df = pd.DataFrame({'t': times, 'x': xs, 'y': ys})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-VT',
        velocity_threshold=10.0, min_fixation_duration=0.0
    )
    assert len(saccades) == 1, "Should detect one saccade"
    assert len(fixations) == 2, "Should detect two fixations"

def test_all_missing_data():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': np.nan, 'y': np.nan})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT'
    )
    assert saccades.empty and fixations.empty, "No events should be detected with all NaN data"

def test_short_fixation():
    times = np.arange(0, 50, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=100.0
    )
    assert fixations.empty, "Fixation should be too short to detect"
    assert saccades.empty, "No saccades should be detected"

def test_long_fixation():
    times = np.arange(0, 2000, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=0.0
    )
    assert len(fixations) == 1, "Should detect one long fixation"
    assert fixations.iloc[0]['Fixation_Duration'] == 1990.0, "Fixation duration should match time span"

def test_no_data_in_window():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', 1000, 2000, algorithm='I-DT'
    )
    assert saccades.empty and fixations.empty, "No events should be detected outside the window"

def test_invalid_algorithm():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    with pytest.raises(ValueError):
        calculate_saccades_and_fixations(
            df, 'x', 'y', 't', times[0], times[-1], algorithm='INVALID'
        )

def test_zero_dispersion_threshold():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT',
        dispersion_threshold=0.0, min_fixation_duration=0.0
    )
    assert fixations.empty, "No fixations should be detected with zero dispersion threshold"

def test_infinite_velocity_threshold():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-VT',
        velocity_threshold=float('inf'), min_fixation_duration=0.0
    )
    assert len(fixations) == 1, "Entire sequence should be one fixation with infinite velocity threshold"
    assert saccades.empty, "No saccades should be detected"

def test_non_numeric_data():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': 'a', 'y': 'b'})
    with pytest.raises(TypeError):
        calculate_saccades_and_fixations(
            df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT'
        )

def test_duplicate_timestamps():
    times = [0, 10, 10, 20]
    df = pd.DataFrame({'t': times, 'x': [100, 100, 100, 100], 'y': [200, 200, 200, 200]})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=0.0
    )
    assert len(fixations) == 1, "Should detect one fixation despite duplicate timestamps"

def test_unsorted_time():
    times = [0, 20, 10, 30]
    df = pd.DataFrame({'t': times, 'x': [100, 100, 100, 100], 'y': [200, 200, 200, 200]})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', min(times), max(times), algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=0.0
    )
    assert len(fixations) == 1, "Should detect one fixation after sorting"

def test_single_point():
    times = [0]
    df = pd.DataFrame({'t': times, 'x': [100], 'y': [200]})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[0], algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=0.0
    )
    assert fixations.empty, "Single point should not form a fixation"
    assert saccades.empty, "Single point should not form a saccade"

def test_fixation_at_start():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0] - 100, times[-1], algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=0.0
    )
    assert len(fixations) == 1, "Should detect one fixation starting at window start"
    assert fixations.iloc[0]['Fixation_StartTime'] == times[0], "Fixation should start at first timestamp"

def test_fixation_at_end():
    times = np.arange(0, 1000, 10)
    df = pd.DataFrame({'t': times, 'x': 100.0, 'y': 200.0})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1] + 100, algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=0.0
    )
    assert len(fixations) == 1, "Should detect one fixation ending at window end"
    assert fixations.iloc[0]['Fixation_EndTime'] == times[-1], "Fixation should end at last timestamp"

def test_saccade_at_start():
    times = np.arange(0, 100, 10)
    xs = [100.0, 200.0] + [200.0]*8
    ys = [200.0, 300.0] + [300.0]*8
    df = pd.DataFrame({'t': times, 'x': xs, 'y': ys})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0] - 10, times[-1], algorithm='I-VT',
        velocity_threshold=10.0, min_fixation_duration=0.0
    )
    assert len(saccades) == 1, "Should detect one saccade at start"
    assert saccades.iloc[0]['Saccade_StartTime'] == times[0], "Saccade should start at first timestamp"

def test_saccade_at_end():
    times = np.arange(0, 100, 10)
    xs = [100.0]*8 + [100.0, 200.0]
    ys = [200.0]*8 + [200.0, 300.0]
    df = pd.DataFrame({'t': times, 'x': xs, 'y': ys})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1] + 10, algorithm='I-VT',
        velocity_threshold=10.0, min_fixation_duration=0.0
    )
    assert len(saccades) == 1, "Should detect one saccade at end"
    assert saccades.iloc[0]['Saccade_EndTime'] == times[-1], "Saccade should end at last timestamp"

def test_high_frequency_noise():
    times = np.arange(0, 100, 10)
    xs = [100 + (-1)**i * 10 for i in range(10)]
    ys = [200 + (-1)**i * 10 for i in range(10)]
    df = pd.DataFrame({'t': times, 'x': xs, 'y': ys})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT',
        dispersion_threshold=20.0, min_fixation_duration=0.0
    )
    assert not fixations.empty or not saccades.empty, "Should detect some events in noisy data"

def test_low_frequency_drift():
    times = np.arange(0, 1000, 10)
    xs = 100 + np.arange(100) * 0.1
    ys = 200 + np.arange(100) * 0.1
    df = pd.DataFrame({'t': times, 'x': xs, 'y': ys})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT',
        dispersion_threshold=50.0, min_fixation_duration=0.0
    )
    assert not fixations.empty, "Should detect at least one fixation in slow drift"

def test_mixed_valid_invalid_data():
    times = np.arange(0, 100, 10)
    xs = [100.0, np.nan, 100.0, np.nan, 100.0, np.nan, 100.0, np.nan, 100.0, np.nan]
    ys = [200.0, np.nan, 200.0, np.nan, 200.0, np.nan, 200.0, np.nan, 200.0, np.nan]
    df = pd.DataFrame({'t': times, 'x': xs, 'y': ys})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-DT',
        dispersion_threshold=1.0, min_fixation_duration=0.0
    )
    assert not fixations.empty, "Should detect fixations between NaN gaps"

def test_extreme_coordinate_values():
    times = np.arange(0, 100, 10)
    xs = [1e6]*5 + [1e-6]*5
    ys = [1e6]*5 + [1e-6]*5
    df = pd.DataFrame({'t': times, 'x': xs, 'y': ys})
    saccades, fixations = calculate_saccades_and_fixations(
        df, 'x', 'y', 't', times[0], times[-1], algorithm='I-VT',
        velocity_threshold=1e5, min_fixation_duration=0.0
    )
    assert len(saccades) == 1, "Should detect one saccade with extreme values"
    assert len(fixations) == 2, "Should detect two fixations with extreme values"