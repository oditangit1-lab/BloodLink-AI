from flask import Blueprint, jsonify, request
from api.utils.data_loader import get_data

donors_bp = Blueprint('donors', __name__)

@donors_bp.route('/donors', methods=['GET'])
def get_donors():
    data = get_data()
    donors = data['donors']
    blood_type = request.args.get('blood_type')
    city = request.args.get('city')
    min_age = request.args.get('min_age', type=int)
    max_age = request.args.get('max_age', type=int)
    # Filter
    if blood_type:
        donors = [d for d in donors if d.get('blood_type') == blood_type]
    if city:
        donors = [d for d in donors if d.get('location', {}).get('city', '').lower() == city.lower()]
    if min_age is not None:
        donors = [d for d in donors if d.get('age', 0) >= min_age]
    if max_age is not None:
        donors = [d for d in donors if d.get('age', 0) <= max_age]
    return jsonify(donors)

@donors_bp.route('/donors/<donor_id>', methods=['GET'])
def get_donor(donor_id):
    data = get_data()
    donor = next((d for d in data['donors'] if d.get('donor_id') == donor_id), None)
    if donor:
        return jsonify(donor)
    return jsonify({'error': 'Donor not found'}), 404