<!-- SPDX-FileCopyrightText: 2025 Tomi Božak, Jožef Stefan Institute -->
<!-- SPDX-License-Identifier: MIT -->
# EyeTrackLib

A research-grade Python toolkit for processing raw eye-tracking data. EyeTrackLib provides robust algorithms to detect:

- **Fixations & Saccades** (I-DT primary, I-VT in beta)
- **Blinks** (time-gap and NaN-run methods)

It also includes a companion visualization app to overlay detected events on gaze plots.

## Installation

```bash
git clone https://github.com/Space0code/EyeTrackLib.git
cd EyeTrackLib
pip install -e .
```

## Quick Start

```python
import pandas as pd
from EyeTrackLib.calculate_streams.saccades_and_fixations import calculate_saccades_and_fixations

df = pd.read_csv("sample_data.csv")  # must have time_ms, gaze_x, gaze_y
saccades, fixations = calculate_saccades_and_fixations(
    df, "gaze_x", "gaze_y", "time_ms",
    window_start_time=0, window_end_time=df["time_ms"].max()
)
```

## Documentation

- **Blink Detection**: see `calculate_streams/README-blinks.md`
- **Fixations & Saccades**: see `calculate_streams/README-fixations-saccades.md`
- **Feature Extraction** (coming soon): code in `calculate_features/`

## Project Structure

```
.
├── AUTHORS.md
├── calculate_features
├── calculate_streams
│   ├── __init__.py
│   ├── blinks.py
│   ├── README-blinks.md
│   ├── README-fixations-saccades.md
│   └── saccades_and_fixations.py
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── paper.md
├── pyproject.toml
├── README.md
└── tests
    └── test_calculate_saccades_and_fixations.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


## License

MIT License — see [LICENSE](LICENSE).


## Citation

If you use EyeTrackLib in your work, please cite it as:

Božak, T. (2025). *EyeTrackLib: An open-source Python toolkit for fixation, saccade, blink detection and feature extraction* (Version 0.9.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.xxxxxx

Alternatively, cite the software directly via GitHub (pre-DOI):

```bibtex
@software{bozak_eyetracklib_2025,
  author  = {Božak, Tomi},
  title   = {EyeTrackLib: An open-source Python toolkit for fixation, saccade, blink detection and feature extraction},
  year    = {2025},
  version = {0.9.0},
  url     = {https://github.com/Space0code/EyeTrackLib}
}
