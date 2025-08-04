from flask import Flask, render_template,request
from flask_socketio import SocketIO
socketio = SocketIO()

from routes.donors import donors_bp
from routes.patients import patients_bp
from routes.matching import matching_bp
from routes.simulate_new_patient import simulate_bp
from routes.training import training_bp
from routes.predict import predict_bp
from routes.matching import get_patient, get_eligible_donors
from routes.predict import predict_availability  
from routes.predict import predict_availability_score
from routes.about import about_bp
from routes.education import education_bp

from datetime import datetime ,timedelta
from utils.data_loader import get_data
import random
from flask import Blueprint, jsonify
import json
import os


def create_app():
    app = Flask(__name__)
    socketio.init_app(app)
    app.register_blueprint(donors_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(matching_bp)
    app.register_blueprint(simulate_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(about_bp)
    app.register_blueprint(education_bp)


    @app.route('/')
    def home():
        return render_template('login.html')

    @app.route('/save_login', methods=['POST'])
    def save_login():
        data = request.get_json()
        try:
            with open('api/static/login.json', 'r', encoding='utf-8') as f:
                logins = json.load(f)
        except Exception:
            logins = []
        logins.append(data)
        with open('api/static/login.json', 'w', encoding='utf-8') as f:
            json.dump(logins, f, ensure_ascii=False, indent=2)
        return jsonify({"status": "saved"})

    @app.route('/dashboard')
    def dashboard():
        all_donors = get_data()['donors']
        count = sum(
            1 for donor in all_donors
            if datetime.strptime(donor['next_eligible'], "%Y-%m-%d") <= datetime.today() + timedelta(days=7)
        )
        return render_template('dashboard.html', donors_available_this_week=count)
        #return render_template('dashboard.html')

    @app.route('/donors_page')
    def donors_page():
        return render_template('donors.html')

    @app.route('/patients_page')
    def patients_page():
        return render_template('patients.html')

    @app.route('/education')
    def education_page():
        return render_template('education.html')

    @app.route('/data')
    def data_page():
        return render_template('data.html')
    
    @app.route('/logindata')
    def logindata():
        return render_template('logindata.html')
    

    @app.route('/chat')
    def chat():
        data = get_data()
        patient = random.choice(data['patients'])
        donors = random.sample(data['donors'], min(10, len(data['donors'])))
        return render_template('chat.html', patient=patient, donors=donors)

    @app.route('/smart_matching/<patient_id>')
    def smart_matching(patient_id):
        patient = get_patient(patient_id)
        results = {k: [] for k in ['manual', 'logreg', 'rf', 'xgb', 'lgbm', 'ensemble']}
        if patient is None:
            return render_template('smart_matching.html', patient_id=patient_id, results=results, error="Patient not found.")
        donors = get_eligible_donors(patient)
        if not donors:
            return render_template('smart_matching.html', patient_id=patient_id, results=results, error="No matching donors found.")
        for model_key in ['manual', 'logreg', 'rf', 'xgb', 'lgbm', 'ensemble']:
            scores = []
            if model_key == 'manual':
                scores = [{'donor': d, 'score': None} for d in donors]
            else:
                for donor in donors:
                    score = predict_availability_score(donor, model_key)  # Direct call, not HTTP
                    scores.append({'donor': donor, 'score': score})
                scores = sorted(scores, key=lambda x: x['score'] or 0, reverse=True)
            results[model_key] = scores

        highly_available = [d for d in results['ensemble'] if d['score'] and d['score'] > 0.8]
        if highly_available:
            # Example: send email, SMS, or just show a message
            notification = f"We found {len(highly_available)} donors with high predicted availability for this patient."
        else:
            notification = ""
        return render_template('smart_matching.html', patient_id=patient_id, results=results, notification=notification)
        # return render_template('smart_matching.html', patient_id=patient_id, results=results)


    from flask import session, redirect, url_for

    @app.route('/logout', methods=['POST'])
    def logout():
        session.clear()
        return '', 204  # No content

    return app
# if __name__ == '__main__':
#     app = create_app()
#     socketio.run(app, debug=True)
if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)