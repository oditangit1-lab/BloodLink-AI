from flask import Blueprint, render_template
import random
from utils.data_loader import get_data

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat')
def chat():
    data = get_data()
    # Get 1 random patient
    patient = random.choice(data['patients'])
    # Get 10 random donors
    donors = random.sample(data['donors'], min(10, len(data['donors'])))
    return render_template('chat.html', patient=patient, donors=donors)