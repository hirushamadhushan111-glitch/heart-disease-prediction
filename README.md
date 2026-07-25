# ❤️ Heart Disease Prediction

A machine learning project that predicts heart disease from 13 clinical
measurements, with a Flask web app for making predictions.

> ⚕️ **Not a medical tool.** This is a student project. It is ~80% accurate
> and will be wrong roughly 1 time in 5. Never use it for a health decision.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 80.3% |
| Precision | 0.786 |
| Recall | 0.786 |
| F1 Score | 0.786 |
| ROC AUC | 0.881 |
| 5-fold CV Accuracy | 82.1% (± 3.2%) |

**Model:** Random Forest (100 trees) on 302 unique patients — 241 train / 61 test.

**Most important features:** `ca` (blocked vessels), `oldpeak` (ST depression),
`cp` (chest pain type), `thalach` (max heart rate), `thal` (thalassemia).

### ⚠️ About the dataset — why not 100%?

`heart.csv` is the Kaggle version of the UCI Cleveland dataset. It looks like
it has **1025 patients**, but **723 of those rows are exact duplicates** — there
are only **302 unique patients**.

Leaving the duplicates in puts the *same patient* in both the training set and
the test set. The model then just memorises the answer instead of learning, and
reports a **fake 100% accuracy**. This is called **data leakage**.

This project removes the duplicates first, so 80.3% is the *honest* score.

The notebook also **asserts** that only 302 patients remain before training, so
the cleaning step can never be skipped by accident.

### Label correction

The Kaggle labels are inverted relative to the original UCI data
(`target=1` means *no* disease there). The notebook flips them and proves the
fix against medical expectations:

| Feature | Disease rate | Expected |
|---|---|---|
| `exang` = 0 → 1 (exercise angina) | 0.305 → 0.768 | should rise ✅ |
| `ca` = 0 → 3 (blocked vessels) | 0.257 → 0.850 | should rise ✅ |
| `sex` female → male | 0.250 → 0.553 | male higher ✅ |

---

## Project structure

```
.
├── heart.csv                          # Dataset (1025 rows, 302 unique)
├── heart_diesease_prdiction.ipynb     # Full analysis notebook (Colab-ready)
├── train_model.py                     # Reproducible training script
├── requirements.txt
└── heart-disease-app/                 # Flask web app
    ├── app.py
    ├── heart_model.pkl                # Trained Random Forest
    ├── heart_scaler.pkl               # StandardScaler (required!)
    ├── model_metrics.json             # Test-set scores shown in the UI
    └── templates/index.html
```

---

## Setup

```bash
pip install -r requirements.txt
```

## Retrain the model

```bash
python train_model.py
```

This removes duplicates, fixes labels, compares 5 algorithms, runs 5-fold
cross-validation and GridSearchCV, then writes `heart_model.pkl`,
`heart_scaler.pkl` and `model_metrics.json` into `heart-disease-app/`.

## Run the web app

```bash
cd heart-disease-app
python app.py
```

Open <http://127.0.0.1:5000>.

| Environment variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `5000` | Port |
| `FLASK_DEBUG` | *(off)* | Set to `1` for the debugger — **local only** |

> The Werkzeug debugger allows arbitrary code execution, so `FLASK_DEBUG`
> is off by default and must never be enabled on a public server.

## Run the notebook

Upload `heart_diesease_prdiction.ipynb` and `heart.csv` to
[Google Colab](https://colab.research.google.com), then choose
**Runtime → Restart session and run all**.

It also runs locally with Jupyter. The Colab-only steps (file upload, Gradio
demo, file download) are skipped automatically outside Colab.

---

## Input features

| Field | Description | Range |
|---|---|---|
| `age` | Age in years | 20–100 |
| `sex` | 0 = female, 1 = male | 0–1 |
| `cp` | Chest pain type (typical / atypical angina, non-anginal, asymptomatic) | 0–3 |
| `trestbps` | Resting blood pressure (mm Hg) | 80–220 |
| `chol` | Serum cholesterol (mg/dl) | 100–600 |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0–1 |
| `restecg` | Resting ECG (normal / ST-T abnormality / LV hypertrophy) | 0–2 |
| `thalach` | Maximum heart rate achieved | 60–220 |
| `exang` | Exercise induced angina | 0–1 |
| `oldpeak` | ST depression induced by exercise | 0.0–7.0 |
| `slope` | Slope of peak exercise ST segment | 0–2 |
| `ca` | Major vessels coloured by fluoroscopy | 0–3 |
| `thal` | Thalassemia (fixed defect / normal / reversible defect) | 1–3 |

---

## Limitations

- Only **302 unique patients** — a small sample, so scores carry real
  uncertainty (hence the ±3.2% cross-validation spread).
- Larger public heart datasets exist (e.g. the 918-row combined set), but they
  **drop `ca` and `thal`** — the 1st and 5th most important features here.
  Tested on this data, removing them lowers cross-validated accuracy from
  **82.1% to 74.8%**, so the 13-feature dataset was kept deliberately.
- The decision threshold is the default 0.5. For a real screening tool a lower
  threshold would raise recall (catching more sick patients) at the cost of
  more false alarms.

## Dataset

UCI Machine Learning Repository — Heart Disease (Cleveland).
<https://archive.ics.uci.edu/dataset/45/heart+disease>
