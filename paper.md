---
title: "EyeTrackLib: A Python toolkit for fixation, saccade, and blink detection from eye-tracking data"
authors:
  - name: Tomi Božak
    orcid: 0009-0002-3560-6419
affiliations:
  - name: Jožef Stefan Institute, Ljubljana, Slovenia
date: 25 July 2025
---

# Summary

EyeTrackLib is an open-source Python toolkit designed for robust detection and analysis of eye movements (fixations, saccades) and blinks using raw gaze coordinate data obtained from eye-tracking systems. The toolkit provides implementations of widely used eye-gaze metrics, enabling researchers in psychology, neuroscience, human–computer interaction (HCI), and related fields to quickly extract meaningful and standardized eye-tracking features. In future releases, EyeTrackLib will also incorporate pupil-size data processing to calculate widely used pupil-size–based metrics, further broadening its research applicability.

Eye movements and blink patterns have proven valuable in various research domains, such as assessing cognitive load, stress, emotional states, and overall wellbeing. Moreover, they are widely utilized in visual-attention studies, including marketing research, usability testing, and general psychological investigations of Areas of Interest (AOIs).

# Statement of need

While other Python packages exist to process eye-tracking data, EyeTrackLib addresses important limitations in currently available libraries. For instance, **PyGazeAnalyser** (Dalmaijer, 2014) offers basic methods for blink, saccade, and fixation detection. EyeTrackLib extends these capabilities substantially by providing more sophisticated algorithms, enhanced robustness, and flexibility:

- **Blink detection**: Unlike PyGazeAnalyser, EyeTrackLib incorporates both minimum and maximum blink-duration thresholds, significantly reducing false positives caused by sensor occlusion or transient artifacts.
- **Saccade detection**: PyGazeAnalyser identifies saccades using inter-sample velocity measured directly in pixels. However, this approach tends to be sensitive and error-prone, particularly at the screen edges, since it ignores the third spatial coordinate required for true angular-velocity calculation. EyeTrackLib instead identifies saccades indirectly by defining them as periods without detected fixations (checking for (in)valid data). This is a more robust and appropriate solution when only two-dimensional data (X, Y coordinates) are available.
- **Fixation detection**: Both PyGazeAnalyser and EyeTrackLib employ the dispersion-threshold (I-DT) algorithm, but EyeTrackLib provides enhancements that improve the robustness and flexibility of fixation detection, including better handling of noisy or missing data.
- **Comparison to EMIP-Toolkit**: EMIP-Toolkit (Al Madi et al., 2021) also implements an I-DT classifier. EyeTrackLib differentiates itself through a clearer API, richer feature exports, and comprehensive documentation for reproducible research.
- **Deep-learning alternatives**: Deep EM Classifier (Startsev et al., 2021) offers high accuracy via CNNs but incurs greater computational cost and reduced interpretability. EyeTrackLib provides explainable, efficient algorithms suitable for most lab and field studies.

# Algorithmic implementation

EyeTrackLib primarily utilizes the **Dispersion Threshold Identification (I-DT)** algorithm for fixation detection. In the I-DT algorithm, fixations are identified as periods where gaze points remain spatially clustered—within a dispersion threshold—for a minimum duration. Saccades are then inferred as the intervals between consecutive fixations. Although the **Velocity Threshold Identification (I-VT)** algorithm is also implemented (in beta), it is best suited for angular-velocity measurements when three-dimensional data are available. For two-dimensional gaze data, I-DT offers greater robustness against corner distortions and missing samples.

# Current use and ongoing research

EyeTrackLib is actively used in the bilateral **TRUST_ME** project (https://dis.ijs.si/trust-me/). One manuscript employing the toolkit is under review at IEEE Transactions on Affective Computing, and another is in submission.

# Future directions

Upcoming releases will integrate **pupil-size** signal analysis, enabling computation of metrics such as average pupil diameter, dilation events, and dynamics related to cognitive load and emotional processing.

# Acknowledgments

This development is supported by the bilateral project TRUST_ME, co-funded by the Swiss National Science Foundation (grant 205121L_214991) and the Slovenian Research and Innovation Agency (grant N1-0319). Shivalika Goyal was supported by a Swiss Government Excellence Scholarship (No. 2024.0108). We also acknowledge support from the Slovenian Research and Innovation Agency research and infrastructure programme (grant P2-0209).

# References

- Dalmaijer, E. S. (2014). PyGazeAnalyser: An open-source Python package for eye-tracking data analysis. Retrieved from https://github.com/esdalmaijer/PyGazeAnalyser  
- Al Madi, N., Guarnera, D., Sharif, B., & Maletic, J. (2021). EMIP toolkit: A Python library for customized post-processing of the eye movements in programming dataset. *ACM Symposium on Eye Tracking Research and Applications*, 1–6.  
- Startsev, M., Agtzidis, I., & Dorr, M. (2021). Deep EM Classifier: A CNN-based classifier for eye movement events. *IEEE Transactions on Neural Networks and Learning Systems*, 33(12), 7373–7384.  