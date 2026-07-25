"""
Train the heart disease model and save heart_model.pkl / heart_scaler.pkl.

Mirrors the notebook pipeline, but runs top-to-bottom in one go so the
duplicate-removal and label-fix steps can never be skipped by accident
(that is what produced the fake 100% accuracy in the old saved model).

Run:  python train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
APP_DIR = HERE / 'heart-disease-app'
RANDOM_STATE = 42

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


def load_data():
    df = pd.read_csv(HERE / 'heart.csv')
    print(f'Raw rows: {len(df)}')

    # --- Remove duplicates (CRITICAL) ---
    # The Kaggle 1025-row file is the 302-patient UCI Cleveland set copied
    # ~3x. Leaving duplicates in puts the SAME patient in both train and
    # test -> the model memorises instead of learning -> fake 100% accuracy.
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f'Duplicates removed: {dups}  ->  {len(df)} unique patients')

    # --- Fix inverted labels ---
    # Verified against the original UCI data: Kaggle target=1 means NO
    # disease. Clinically confirmed below (exang / ca / sex disease rates).
    df['target'] = 1 - df['target']
    print(f"Class balance: {df['target'].value_counts().to_dict()}")

    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    return df


def sanity_check(df):
    """Prove the label flip is correct using clinical expectations."""
    print('\n--- Clinical sanity check (mean disease rate) ---')
    for col, note in [('exang', 'exercise angina -> should be HIGHER'),
                      ('ca', 'more blocked vessels -> should RISE'),
                      ('sex', '1=male -> should be HIGHER')]:
        rates = df.groupby(col)['target'].mean().round(3).to_dict()
        print(f'  {col:6s} {rates}   ({note})')


def compare_models(X_train, X_test, y_train, y_test, X, y):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, trained = [], {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        # Scaler re-fitted inside each fold so CV has no leakage either.
        cv_scores = cross_val_score(make_pipeline(StandardScaler(), model),
                                    X, y, cv=cv, scoring='accuracy')
        rows.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred),
            'F1': f1_score(y_test, pred),
            'AUC': roc_auc_score(y_test, prob),
            'CV Acc': cv_scores.mean(),
            'CV Std': cv_scores.std(),
        })

    table = pd.DataFrame(rows).sort_values('F1', ascending=False)
    print('\n--- Model comparison (held-out test set, 5-fold CV) ---')
    print(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    return table, trained


def tune_random_forest(X_train, y_train):
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {'n_estimators': [50, 100, 200],
         'max_depth': [3, 5, 10, None],
         'min_samples_split': [2, 5, 10]},
        cv=5, scoring='f1', n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f'\nGridSearch best params : {grid.best_params_}')
    print(f'GridSearch best CV F1  : {grid.best_score_:.3f}')
    return grid.best_estimator_


def main():
    df = load_data()
    sanity_check(df)

    X, y = df[FEATURES], df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f'\nTrain: {len(X_train)}   Test: {len(X_test)}')

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)

    table, trained = compare_models(X_train_s, X_test_s, y_train, y_test, X, y)

    best_name = table.iloc[0]['Model']
    best_model = trained[best_name]
    best_f1 = table.iloc[0]['F1']

    # Only promote the tuned forest if it actually beats the winner.
    tuned = tune_random_forest(X_train_s, y_train)
    tuned_f1 = f1_score(y_test, tuned.predict(X_test_s))
    print(f'Tuned RF test F1       : {tuned_f1:.3f}  (current best {best_f1:.3f})')
    if tuned_f1 > best_f1:
        best_model, best_name, best_f1 = tuned, 'Random Forest (tuned)', tuned_f1
        print('-> Tuned Random Forest promoted to best model.')
    else:
        print(f'-> Keeping {best_name}.')

    pred = best_model.predict(X_test_s)
    print(f'\n=== FINAL MODEL: {best_name} ===')
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred,
                                target_names=['No Disease', 'Disease']))

    importances = pd.Series(best_model.feature_importances_,
                            index=FEATURES).sort_values(ascending=False)
    print('Top 5 features:', importances.head(5).round(3).to_dict())

    APP_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, APP_DIR / 'heart_model.pkl')
    joblib.dump(scaler, APP_DIR / 'heart_scaler.pkl')

    metrics = {
        'model': best_name,
        'unique_patients': len(df),
        'train_n': len(X_train), 'test_n': len(X_test),
        'accuracy': round(accuracy_score(y_test, pred), 3),
        'precision': round(precision_score(y_test, pred), 3),
        'recall': round(recall_score(y_test, pred), 3),
        'f1': round(best_f1, 3),
        'auc': round(roc_auc_score(y_test, best_model.predict_proba(X_test_s)[:, 1]), 3),
        'top_features': importances.head(5).round(3).to_dict(),
    }
    (APP_DIR / 'model_metrics.json').write_text(json.dumps(metrics, indent=2))

    print(f'\nSaved -> {APP_DIR / "heart_model.pkl"}')
    print(f'Saved -> {APP_DIR / "heart_scaler.pkl"}')
    print(f'Saved -> {APP_DIR / "model_metrics.json"}')


if __name__ == '__main__':
    main()
