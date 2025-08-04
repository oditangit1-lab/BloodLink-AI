from flask import Blueprint, jsonify, request
from api.utils.data_loader import get_data

patients_bp = Blueprint('patients', __name__)

@patients_bp.route('/patients', methods=['GET'])
def get_patients():
    data = get_data()
    patients = data['patients']
    # Get query params
    blood_type = request.args.get('blood_type')
    city = request.args.get('city')
    urgency = request.args.get('urgency')
    severity = request.args.get('severity')
    # Filter
    if blood_type:
        patients = [p for p in patients if p.get('blood_type') == blood_type]
    if city:
        patients = [p for p in patients if p.get('location', {}).get('city', '').lower() == city.lower()]
    if urgency:
        patients = [p for p in patients if p.get('urgency_level', '').lower() == urgency.lower()]
    if severity:
        patients = [p for p in patients if p.get('severity', '').lower() == severity.lower()]
    return jsonify(patients)

@patients_bp.route('/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    data = get_data()
    patient = next((p for p in data['patients'] if p.get('patient_id') == patient_id), None)
    if patient:
        return jsonify(patient)
    return jsonify({'error': 'Patient not found'}), 404