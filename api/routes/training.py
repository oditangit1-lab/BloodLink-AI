from flask import Blueprint, render_template, jsonify
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import request
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

training_bp = Blueprint('training', __name__)

def donor_to_features(d):
    today = datetime.today()
    last_donation = pd.to_datetime(d.get('last_donation', None))
    next_eligible = pd.to_datetime(d.get('next_eligible', None))
    days_since_last = (today - last_donation).days if pd.notnull(last_donation) else np.nan
    days_until_next = (next_eligible - today).days if pd.notnull(next_eligible) else np.nan
    # Target: available if next_eligible is in the past
    available = int(pd.isnull(next_eligible) or next_eligible <= today)
    # For demo: randomly set some as unavailable
    import random
    if random.random() < 0.2:
        available = 0
    # Feature engineering: interaction features
    response_reliability = (d.get("availability_pattern", {}).get("response_rate") or 0) * (d.get("behavioral_patterns", {}).get("reliability_score") or 0)
    return {
        "age": d.get("age"),
        "gender": d.get("gender"),
        "blood_type": d.get("blood_type"),
        "city": d.get("location", {}).get("city"),
        "response_rate": d.get("availability_pattern", {}).get("response_rate"),
        "avg_response_time_hours": d.get("availability_pattern", {}).get("avg_response_time_hours"),
        "weekend_availability": int(d.get("availability_pattern", {}).get("weekend_availability", False)),
        "emergency_availability": int(d.get("availability_pattern", {}).get("emergency_availability", False)),
        "reliability_score": d.get("behavioral_patterns", {}).get("reliability_score"),
        "social_influence_score": d.get("behavioral_patterns", {}).get("social_influence_score"),
        "total_donations": d.get("engagement_metrics", {}).get("total_donations"),
        "total_cancellations": d.get("engagement_metrics", {}).get("total_cancellations"),
        "avg_donation_interval_days": d.get("engagement_metrics", {}).get("avg_donation_interval_days"),
        "bmi": d.get("health_profile", {}).get("bmi"),
        "hemoglobin_level": d.get("health_profile", {}).get("hemoglobin_level"),
        "days_since_last_donation": days_since_last,
        "days_until_next_eligible": days_until_next,
        "response_reliability": response_reliability,
        "available": available
    }

@training_bp.route('/model_training')
def model_training():
    # Check if models exist
    model_dir = "models"
    models = {
        "logreg": os.path.exists(os.path.join(model_dir, "model_logistic_regression.pkl")),
        "rf": os.path.exists(os.path.join(model_dir, "model_random_forest.pkl")),
        "xgb": os.path.exists(os.path.join(model_dir, "model_xgboost.pkl")),
        "lgbm": os.path.exists(os.path.join(model_dir, "model_lightgbm.pkl")),
        "ensemble": os.path.exists(os.path.join(model_dir, "model_ensemble.pkl")),
    }
    existing_models = [name for name, exists in models.items() if exists]
    return render_template('model_training.html', models=models, existing_models=existing_models)

@training_bp.route('/train_models', methods=['POST'])
def train_models():
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    req = request.get_json()
    model_key = req.get('model')

    # Load donor data
    with open('api/data/data.json', 'r') as f:
        data = json.load(f)
        donors = data['donors']
        df = pd.DataFrame([donor_to_features(d) for d in donors])

    features = [
        "age", "gender", "blood_type", "city", "response_rate", "avg_response_time_hours",
        "weekend_availability", "emergency_availability", "reliability_score", "social_influence_score",
        "total_donations", "total_cancellations", "avg_donation_interval_days", "bmi", "hemoglobin_level",
        "days_since_last_donation", "days_until_next_eligible"
    ]
    X = df[features]
    y = df['available']
    X = pd.get_dummies(X)

    # Balance classes
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

    model_map = {
        "logreg": LogisticRegression(max_iter=3000, class_weight='balanced'),
        "rf": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "lgbm": LGBMClassifier()
    }
    model_names = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost",
        "lgbm": "LightGBM"
    }

    if model_key not in model_map:
        return jsonify({"status": "error", "message": "Invalid model key."})

    model = model_map[model_key]
    model_name = model_names[model_key]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:,1]
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    fname = f"models/model_{model_name.replace(' ','_').lower()}.pkl"
    joblib.dump(model, fname)

    return jsonify({
        "status": "done",
        "accuracy": round(acc,3),
        "auc": round(auc,3),
        "file": fname
    })


@training_bp.route('/train_ensemble', methods=['POST'])
def train_ensemble():
    import joblib
    from sklearn.metrics import accuracy_score, roc_auc_score
    import pandas as pd
    import numpy as np

    # Load your donor data from JSON
    with open('api/data/data.json', 'r') as f:
        data = json.load(f)
        donors = data['donors']
        df = pd.DataFrame([donor_to_features(d) for d in donors])

    features = [
        "age", "gender", "blood_type", "city", "response_rate", "avg_response_time_hours",
        "weekend_availability", "emergency_availability", "reliability_score", "social_influence_score",
        "total_donations", "total_cancellations", "avg_donation_interval_days", "bmi", "hemoglobin_level",
        "days_since_last_donation", "days_until_next_eligible"
    ]
    X = df[features]
    y = df['available']
    X = pd.get_dummies(X)

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

    # Load trained models
    rf_model = joblib.load("models/model_random_forest.pkl")
    xgb_model = joblib.load("models/model_xgboost.pkl")
    lgbm_model = joblib.load("models/model_lightgbm.pkl")
    logreg_model = joblib.load("models/model_logistic_regression.pkl")

    # Get predictions
    y_pred_rf = rf_model.predict_proba(X_val)[:,1]
    y_pred_xgb = xgb_model.predict_proba(X_val)[:,1]
    y_pred_lgbm = lgbm_model.predict_proba(X_val)[:,1]
    y_pred_logreg = logreg_model.predict_proba(X_val)[:,1]

    # Average probabilities
    ensemble_pred = (y_pred_rf + y_pred_xgb + y_pred_lgbm + y_pred_logreg) / 4
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    ensemble_acc = accuracy_score(y_val, (ensemble_pred > 0.5).astype(int))

    joblib.dump({'ensemble_auc': ensemble_auc, 'ensemble_acc': ensemble_acc}, "models/model_ensemble.pkl")

    return jsonify({
        "status": "done",
        "accuracy": round(ensemble_acc,3),
        "auc": round(ensemble_auc,3),
        "file": "models/model_ensemble.pkl"
    })
 