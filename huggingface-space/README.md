---
title: Heart Disease Prediction
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Predict heart disease from 13 clinical measurements
---

# ❤️ Heart Disease Prediction

Predicts heart disease from 13 clinical measurements using a Random Forest
trained on the UCI Cleveland dataset.

> ⚕️ **Not a medical tool.** A student project, ~80% accurate. It will be wrong
> roughly 1 time in 5. Always consult a doctor.

| Metric | Score |
|---|---|
| Accuracy | 80.3% |
| Recall | 0.786 |
| ROC AUC | 0.881 |
| 5-fold CV | 82.1% (± 3.2%) |

## Why not 100%?

The Kaggle version of this dataset looks like it has 1025 patients, but **723
rows are exact duplicates** — there are only **302 unique patients**. Leaving
them in puts the same patient in both the training and test sets, so the model
memorises the answers and reports a fake 100% accuracy (**data leakage**).

This project removes the duplicates first, so 80.3% is the honest score.

## Source

Full project, Flask app and analysis notebook:
<https://github.com/hirushamadhushan111-glitch/heart-disease-prediction>
