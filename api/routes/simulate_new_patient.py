import random
from flask import Blueprint, jsonify, current_app
from flask_socketio import emit
from api.utils.data_loader import get_data

simulate_bp = Blueprint('simulate', __name__)

@simulate_bp.route('/simulate_new_patient', methods=['POST'])
def simulate_new_patient():
    data = get_data()
    patients = data['patients']
    # Pick 1-3 random patients
    new_patients = random.sample(patients, k=random.randint(1, 3))
    socketio = current_app.extensions['socketio']  # Get the socketio instance
    for patient in new_patients:
        socketio.emit('new_patient', patient)
    return jsonify({"status": "ok", "count": len(new_patients)})