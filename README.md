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

Five algorithms were compared — Logistic Regression, Decision Tree, Random
Forest, K-Nearest Neighbors and Naive Bayes — and evaluated on a held-out test
set plus 5-fold cross-validation. Random Forest won on F1 score and was tuned
with `GridSearchCV`.

---

## Project structure

```
.
├── heart.csv                          # UCI Cleveland dataset
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
python train_model.py            # full run: 5 models, CV, grid search (~1 min)
python train_model.py --quick    # winning Random Forest only (~2 seconds)
```

Either way this writes `heart_model.pkl`, `heart_scaler.pkl` and
`model_metrics.json` into `heart-disease-app/`.

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
