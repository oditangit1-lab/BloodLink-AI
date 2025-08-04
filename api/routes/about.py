from flask import Blueprint, jsonify

about_bp = Blueprint('about', __name__)

def load_qa():
    qa = []
    with open('api/data/about.txt', encoding='utf-8') as f:
        lines = f.readlines()
    question, answer = None, ""
    for line in lines:
        if line.startswith("Q"):
            if question:
                qa.append({"question": question, "answer": answer.strip()})
            question = line.strip().split(":", 1)[1].strip()
            answer = ""
        elif line.startswith("A:"):
            answer = line.strip().split(":", 1)[1].strip()
        else:
            answer += " " + line.strip()
    if question:
        qa.append({"question": question, "answer": answer.strip()})
    return qa

@about_bp.route('/about_qa')
def about_qa():
    return jsonify(load_qa())