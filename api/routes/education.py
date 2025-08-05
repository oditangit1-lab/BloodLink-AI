import requests
from flask import Blueprint, request, jsonify
import difflib
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='api/.env')
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
education_bp = Blueprint('education', __name__)

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

def get_context():
    with open('data/education.txt', encoding='utf-8') as f:
        return f.read()

@education_bp.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip().lower()
    if not question:
        return jsonify({"answer": "Please ask a question."})

    # Load and split education context
    with open('data/education.txt', encoding='utf-8') as f:
        paragraphs = [p.strip() for p in f.read().split('\n\n') if p.strip()]

    # Use simple scoring: find paragraphs with the most overlap
    best_match = ""
    highest_score = 0

    for para in paragraphs:
        score = sum(word in para.lower() for word in question.split())
        if score > highest_score:
            highest_score = score
            best_match = para

    if highest_score == 0:
        return jsonify({"answer": "Sorry, no matching info found."})
    
    # Trim to words max
    trimmed = ' '.join(best_match.split()[:400])
    return jsonify({"answer": trimmed})

# @education_bp.route('/ask', methods=['POST'])
# def ask():
#     data = request.get_json()
#     question = data.get('question')
#     context = get_context()
#     headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
#     payload = {"question": question, "context": context}
#     response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
#     answer = response.json().get('answer', 'Sorry, no answer found.')
#     return jsonify({"answer": answer})

@education_bp.route('/about_text')
def about_text():
    with open('data/about.txt', encoding='utf-8') as f:return f.read()

@education_bp.route('/education_suggestions')
def education_suggestions():
    with open('data/education.txt', encoding='utf-8') as f:
        text = f.read()
    # Split into sentences or paragraphs
    # For paragraphs:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return jsonify(paragraphs)


