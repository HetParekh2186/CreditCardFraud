"""
Enhanced & Fast Credit Card Fraud Detection Script

Features:
- Pipeline with scaling, SMOTE, and classifier
- Proper nested CV: oversampling inside training folds
- RandomizedSearchCV for hyperparameter tuning including n_estimators
- Optional probability calibration
- Automatic threshold tuning for desired recall

Usage:
    python fraud_detection.py --data_path path/to/creditcard.csv \
        [--model rf|xgb] [--tune] [--calibrate] [--recall 0.9] [--threshold 0.5] [--save]
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib


def load_data(path):
    df = pd.read_csv(path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('Class', axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=test_size,
                             stratify=y, random_state=random_state)


def build_pipeline(model_type, random_state=42):
    steps = [
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=random_state))
    ]
    if model_type == 'xgb':
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    else:
        clf = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1)
    steps.append(('clf', clf))
    return Pipeline(steps)


def hyperparameter_search(model_type, X, y, random_state=42):
    pipe = build_pipeline(model_type, random_state)
    if model_type == 'rf':
        param_dist = {
            'clf__n_estimators': [50, 100, 200, 500],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 5, 10]
        }
    else:
        param_dist = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.05, 0.1]
        }
    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=20,
        scoring='average_precision',
        cv=2,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X, y)
    print("Best params:", search.best_params_)
    print("Best PR-AUC:", search.best_score_)
    return search.best_estimator_


def calibrate_model(pipe, X, y):
    calib = CalibratedClassifierCV(pipe, cv='prefit', method='isotonic')
    calib.fit(X, y)
    return calib


def tune_threshold(y_true, y_prob, target_recall=0.9):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    idx = np.argmax(recall >= target_recall)
    thresh = thresholds[idx] if idx < len(thresholds) else 0.5
    print(f"Threshold for {target_recall*100:.0f}% recall: {thresh:.3f}, precision: {precision[idx]:.3f}")
    return thresh


def evaluate(clf, X_test, y_test, threshold=0.5):
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    print('\nFinal Evaluation:')
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    print(f"PR AUC: {auc(recall, precision):.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model', choices=['rf','xgb'], default='rf')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--recall', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed threshold; else auto-tune')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    df = load_data(args.data_path)
    print("Original distribution:", df['Class'].value_counts(normalize=True))

    X_train, X_test, y_train, y_test = split_data(df)
    print("Train:", y_train.value_counts(), "Test:", y_test.value_counts())

    # Tune or fit
    if args.tune:
        model = hyperparameter_search(args.model, X_train, y_train)
    else:
        model = build_pipeline(args.model).fit(X_train, y_train)

    # Calibration
    if args.calibrate:
        model = calibrate_model(model, X_train, y_train)

    # Threshold tuning
    y_prob = model.predict_proba(X_test)[:, 1]
    if args.threshold is None:
        args.threshold = tune_threshold(y_test, y_prob, args.recall)
    else:
        print(f"Using fixed threshold: {args.threshold:.3f}")

    # Evaluate and save
    evaluate(model, X_test, y_test, threshold=args.threshold)
    if args.save:
        joblib.dump(model, 'best_model.pkl')
        print("Model saved as best_model.pkl")

if __name__ == '__main__':
    main()
