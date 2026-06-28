import os
import sys
from flask import Flask, jsonify, request, render_template

# Add current directory to path to ensure modules are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import model
import train_model

app = Flask(__name__)

# Auto-train models on startup if model.pkl is missing
if not os.path.exists(model.MODEL_PATH):
    try:
        print("Model file not found. Training models on startup...")
        train_model.train_and_save_models()
        print("Startup training completed.")
    except Exception as e:
        print(f"Warning: Failed to train models on startup: {e}")

@app.route('/')
def home():
    """Serves the main application page."""
    return render_template('index.html')

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Returns the list of unique symptoms available for selection and their counts."""
    try:
        symptoms, counts = model.get_symptoms_with_counts()
        # Clean symptoms to be human-readable for frontend (e.g. replacing underscores with spaces, capitalizing)
        formatted_symptoms = [
            {
                'id': s,
                'name': s.replace('_', ' ').title(),
                'count': int(counts.get(s, 0))
            } for s in symptoms
        ]
        # Sort by human readable name
        formatted_symptoms = sorted(formatted_symptoms, key=lambda x: x['name'])
        return jsonify({
            'success': True,
            'symptoms': formatted_symptoms
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predicts a disease based on a list of symptoms and a chosen model."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided.'}), 400
            
        selected_symptoms = data.get('symptoms', [])
        model_type = data.get('model', 'decision_tree')
        
        if not isinstance(selected_symptoms, list):
            return jsonify({'success': False, 'error': 'Symptoms must be a list.'}), 400
            
        if len(selected_symptoms) == 0:
            return jsonify({'success': False, 'error': 'Please select at least one symptom.'}), 400
            
        prediction = model.predict_disease(selected_symptoms, model_type)
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except FileNotFoundError as fnf:
        return jsonify({
            'success': False,
            'error': 'The classification models have not been trained yet. Please retrain models from the Model Information panel.'
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Retrains the ML models on the CSV dataset and returns updated accuracies."""
    try:
        bundle = train_model.train_and_save_models()
        
        # Prepare accuracy statistics for JSON response
        accuracies = {}
        for m_name, m_info in bundle['models'].items():
            accuracies[m_name] = {
                'name': m_name.replace('_', ' ').title(),
                'train_acc': m_info['train_acc'],
                'test_acc': m_info['test_acc']
            }
            
        return jsonify({
            'success': True,
            'message': 'Models successfully retrained.',
            'accuracies': accuracies
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrain models: {str(e)}'
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Returns AI clinical assistant replies for user text messages."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided.'}), 400
            
        user_message = data.get('message', '')
        response_text = model.get_chat_response(user_message)
        
        return jsonify({
            'success': True,
            'reply': response_text
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 200  # Fallback for client routing

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
