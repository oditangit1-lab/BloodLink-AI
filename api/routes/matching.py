from flask import Blueprint, jsonify, request
from api.utils.data_loader import get_data
from datetime import datetime

matching_bp = Blueprint('matching', __name__)

def get_patient(patient_id):
    data = get_data()
    for patient in data['patients']:
        # Try all possible keys and handle case sensitivity
        for key in ['id', 'ID', 'patient_id', 'Patient_ID']:
            if str(patient.get(key, '')).strip() == str(patient_id).strip():
                return patient
    return None

def get_eligible_donors(patient):
    data = get_data()
    donors = data['donors']
    # Match by blood type and city
    matches = [d for d in donors if d.get('blood_type') == patient.get('blood_type')]
    matches = [d for d in matches if d.get('location', {}).get('city', '').lower() == patient.get('location', {}).get('city', '').lower()]
    return matches

@matching_bp.route('/matching', methods=['GET'])
def match_donors():
    blood_type = request.args.get('blood_type')
    city = request.args.get('city')
    urgency = request.args.get('urgency')
    data = get_data()
    donors = data['donors']
    # Basic matching: by blood type and city
    matches = donors
    if blood_type:
        matches = [d for d in matches if d.get('blood_type') == blood_type]
    if city:
        matches = [d for d in matches if d.get('location', {}).get('city', '').lower() == city.lower()]
    # Optionally, add urgency or other filters
    return jsonify(matches)

@matching_bp.route('/matching_with_prediction', methods=['POST'])
def matching_with_prediction():
    data = request.get_json()
    patient = data['patient']
    donors = data['donors']
    model_key = data.get('model', 'ensemble')
    scores = []
    for donor in donors:
        # Call local prediction endpoint
        resp = requests.post(
            request.host_url + 'predict_availability',
            json={'donor': donor, 'model': model_key}
        )
        score = resp.json()['availability_score']
        scores.append({'donor': donor, 'score': score})
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    return jsonify(sorted_scores)