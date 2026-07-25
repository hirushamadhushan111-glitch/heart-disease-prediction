import json
import os
from pathlib import Path

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

HERE = Path(__file__).parent

# Load the trained model and scaler (loads ONCE when server starts)
try:
    model = joblib.load(HERE / 'heart_model.pkl')
    scaler = joblib.load(HERE / 'heart_scaler.pkl')
except (OSError, EOFError) as exc:
    raise SystemExit(
        f'Could not load model files: {exc}\n'
        'Run "python train_model.py" from the parent folder first.'
    ) from exc

# Column order must match the order the scaler/model were trained on
FEATURES = list(scaler.feature_names_in_)

# Test-set metrics written by train_model.py, shown in the footer
try:
    METRICS = json.loads((HERE / 'model_metrics.json').read_text())
except (OSError, ValueError):
    METRICS = None

# Valid ranges for every input field: (min, max, label shown in error message)
RANGES = {
    'age'      : (20, 100,  'Age'),
    'sex'      : (0, 1,     'Sex'),
    'cp'       : (0, 3,     'Chest Pain Type'),
    'trestbps' : (80, 220,  'Resting Blood Pressure'),
    'chol'     : (100, 600, 'Cholesterol'),
    'fbs'      : (0, 1,     'Fasting Blood Sugar'),
    'restecg'  : (0, 2,     'Resting ECG'),
    'thalach'  : (60, 220,  'Max Heart Rate'),
    'exang'    : (0, 1,     'Exercise Induced Angina'),
    'oldpeak'  : (0.0, 7.0, 'ST Depression'),
    'slope'    : (0, 2,     'Slope'),
    'ca'       : (0, 3,     'Major Vessels'),
    'thal'     : (1, 3,     'Thalassemia'),
}

DEFAULTS = {
    'age': '50', 'sex': '1', 'cp': '0', 'trestbps': '120', 'chol': '240',
    'fbs': '0', 'restecg': '1', 'thalach': '150', 'exang': '0',
    'oldpeak': '1.0', 'slope': '1', 'ca': '0', 'thal': '2',
}


@app.route('/')
def home():
    return render_template('index.html', form=DEFAULTS, metrics=METRICS)


@app.route('/predict', methods=['POST'])
def predict():
    # Keep whatever the user typed so the form can be re-filled either way
    form = {field: request.form.get(field, '').strip() for field in RANGES}

    # Validate every form value BEFORE predicting
    values = {}
    for field, (low, high, label) in RANGES.items():
        raw = form[field]
        try:
            value = float(raw)
        except ValueError:
            return render_template(
                'index.html', form=form, metrics=METRICS,
                error=f'❌ {label}: "{raw}" is not a valid number.')
        if not (low <= value <= high):
            return render_template(
                'index.html', form=form, metrics=METRICS,
                error=f'❌ {label} must be between {low} and {high} (got {raw}).')
        values[field] = [value]

    patient = pd.DataFrame(values)[FEATURES]

    # Apply the SAME scaling used in training. transform() returns a plain
    # array, so rebuild the DataFrame - the model was fitted with feature
    # names and warns if it gets an unnamed array.
    patient_scaled = pd.DataFrame(scaler.transform(patient), columns=FEATURES)

    # Predict
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]

    if prediction == 1:
        result = f'⚠️ HEART DISEASE DETECTED ({probability[1] * 100:.1f}% confidence)'
    else:
        result = f'✅ NO HEART DISEASE ({probability[0] * 100:.1f}% confidence)'

    return render_template('index.html', form=form, metrics=METRICS,
                           result=result, risk=round(probability[1] * 100, 1))


if __name__ == '__main__':
    # Never hard-code debug=True: the Werkzeug debugger lets anyone who can
    # reach the server run arbitrary code. Opt in locally with FLASK_DEBUG=1.
    app.run(host=os.environ.get('HOST', '127.0.0.1'),
            port=int(os.environ.get('PORT', 5000)),
            debug=os.environ.get('FLASK_DEBUG') == '1')
