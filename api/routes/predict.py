from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

predict_bp = Blueprint('predict', __name__)


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


def predict_availability_score(donor, model_key):
    import joblib
    import pandas as pd

    df = pd.DataFrame([donor_to_features(donor)])
    X = pd.get_dummies(df)

    # Load model
    model_map = {
        'logreg': 'models/model_logistic_regression.pkl',
        'rf': 'models/model_random_forest.pkl',
        'xgb': 'models/model_xgboost.pkl',
        'lgbm': 'models/model_lightgbm.pkl'
    }
    if model_key == 'ensemble':
        models = [joblib.load(model_map[m]) for m in ['rf', 'xgb', 'lgbm', 'logreg']]
        # Get expected features from one model (all should match)
        expected_features = models[0].feature_names_in_ if hasattr(models[0], 'feature_names_in_') else X.columns
        # Align columns
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        preds = [model.predict_proba(X)[:,1][0] for model in models]
        score = float(sum(preds)/len(preds))
    else:
        model = joblib.load(model_map[model_key])
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X.columns
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        score = float(model.predict_proba(X)[:,1][0])
    return score


@predict_bp.route('/predict_availability', methods=['POST'])
def predict_availability():
    data = request.get_json()
    donor = data['donor']
    model_key = data.get('model', 'ensemble')
    df = pd.DataFrame([donor_to_features(donor)])
    X = pd.get_dummies(df)

    # Load model(s)
    if model_key == 'ensemble':
        rf = joblib.load('models/model_random_forest.pkl')
        xgb = joblib.load('models/model_xgboost.pkl')
        lgbm = joblib.load('models/model_lightgbm.pkl')
        logreg = joblib.load('models/model_logistic_regression.pkl')
        preds = [
            rf.predict_proba(X)[:,1],
            xgb.predict_proba(X)[:,1],
            lgbm.predict_proba(X)[:,1],
            logreg.predict_proba(X)[:,1]
        ]
        score = float(sum(preds)/len(preds))
    else:
        model_map = {
            'logreg': 'models/model_logistic_regression.pkl',
            'rf': 'models/model_random_forest.pkl',
            'xgb': 'models/model_xgboost.pkl',
            'lgbm': 'models/model_lightgbm.pkl'
        }
        model = joblib.load(model_map[model_key])
        score = predict_availability_score(donor, model_key)
    return jsonify({'availability_score': score})