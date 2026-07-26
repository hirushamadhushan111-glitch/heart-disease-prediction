"""
Gradio demo of the heart disease model, for Hugging Face Spaces.

The real project is the Flask app in heart-disease-app/. This is the same
model behind a Gradio UI, because Spaces hosts Gradio for free.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

HERE = Path(__file__).parent

model = joblib.load(HERE / 'heart_model.pkl')
scaler = joblib.load(HERE / 'heart_scaler.pkl')

# Column order must match what the scaler/model were trained on
FEATURES = list(scaler.feature_names_in_)

try:
    METRICS = json.loads((HERE / 'model_metrics.json').read_text())
except (OSError, ValueError):
    METRICS = {}

DISCLAIMER = """
### ⚕️ Not a medical diagnosis

This is a student machine-learning project trained on **302 patient records**
from the UCI Cleveland dataset. It is about **80% accurate** and will be wrong
roughly **1 time in 5**. Never use it to make a health decision — always
consult a qualified doctor.
"""


def predict(age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal):
    patient = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg],
        'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
        'slope': [slope], 'ca': [ca], 'thal': [thal],
    })[FEATURES]

    # transform() returns a plain array; rebuild the frame so the model gets
    # the feature names it was fitted with
    scaled = pd.DataFrame(scaler.transform(patient), columns=FEATURES)

    probability = model.predict_proba(scaled)[0]
    risk = float(probability[1])

    verdict = ('⚠️ **HEART DISEASE DETECTED**' if model.predict(scaled)[0] == 1
               else '✅ **NO HEART DISEASE**')
    summary = f'{verdict}\n\nEstimated risk: **{risk * 100:.1f}%**'

    # Gradio Label component wants {class: probability}
    return summary, {'Disease': risk, 'No Disease': 1 - risk}


# Radio choices as (visible label, value sent to predict)
SEX = [('Female', 0), ('Male', 1)]
CP = [('0 - Typical Angina', 0), ('1 - Atypical Angina', 1),
      ('2 - Non-anginal Pain', 2), ('3 - Asymptomatic', 3)]
YES_NO = [('No', 0), ('Yes', 1)]
RESTECG = [('0 - Normal', 0), ('1 - ST-T Abnormality', 1),
           ('2 - LV Hypertrophy', 2)]
SLOPE = [('0 - Upsloping', 0), ('1 - Flat', 1), ('2 - Downsloping', 2)]
THAL = [('1 - Fixed Defect', 1), ('2 - Normal', 2), ('3 - Reversible Defect', 3)]

# No theme= here on purpose: Gradio 5 wants it on Blocks and Gradio 6 wants it
# on launch(), so passing it either way breaks on the other version. The
# default theme is fine and works everywhere.
with gr.Blocks(title='Heart Disease Prediction') as demo:
    gr.Markdown('# ❤️ Heart Disease Prediction')
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            gr.Markdown('#### Patient details')
            age = gr.Slider(20, 100, value=50, step=1, label='Age')
            sex = gr.Radio(SEX, value=1, label='Sex')
            cp = gr.Radio(CP, value=0, label='Chest Pain Type')
            trestbps = gr.Slider(80, 220, value=120, step=1,
                                 label='Resting Blood Pressure (mm Hg)')
            chol = gr.Slider(100, 600, value=240, step=1,
                             label='Cholesterol (mg/dl)')
            fbs = gr.Radio(YES_NO, value=0,
                           label='Fasting Blood Sugar > 120 mg/dl')
            restecg = gr.Radio(RESTECG, value=1, label='Resting ECG')

        with gr.Column():
            gr.Markdown('#### Exercise test')
            thalach = gr.Slider(60, 220, value=150, step=1,
                                label='Max Heart Rate Achieved')
            exang = gr.Radio(YES_NO, value=0,
                             label='Exercise Induced Angina')
            oldpeak = gr.Slider(0.0, 7.0, value=1.0, step=0.1,
                                label='ST Depression (oldpeak)')
            slope = gr.Radio(SLOPE, value=1, label='Slope of Peak Exercise ST')
            ca = gr.Radio([0, 1, 2, 3], value=0,
                          label='Major Vessels Coloured (ca)')
            thal = gr.Radio(THAL, value=2, label='Thalassemia (thal)')

    button = gr.Button('Predict', variant='primary')
    result = gr.Markdown()
    chart = gr.Label(label='Probability', num_top_classes=2)

    inputs = [age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]
    button.click(predict, inputs=inputs, outputs=[result, chart])

    gr.Examples(
        label='Try an example patient',
        examples=[
            # Low risk: young, high max heart rate, no angina, clear vessels
            [45, 0, 2, 120, 200, 0, 1, 170, 0, 0.0, 2, 0, 2],
            # High risk: older male, exercise angina, 3 blocked vessels
            [67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 3],
        ],
        inputs=inputs,
    )

    if METRICS:
        gr.Markdown(
            f"---\n**Model:** {METRICS.get('model')} · trained on "
            f"{METRICS.get('unique_patients')} unique patients · "
            f"test accuracy {METRICS.get('accuracy', 0) * 100:.1f}% · "
            f"recall {METRICS.get('recall', 0) * 100:.1f}% · "
            f"AUC {METRICS.get('auc')}  \n"
            "[Source code on GitHub]"
            "(https://github.com/hirushamadhushan111-glitch/heart-disease-prediction)"
        )

if __name__ == '__main__':
    demo.launch()
