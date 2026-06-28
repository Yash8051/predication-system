import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Metadata mapping predicted diseases to suggested specialists and precautions
DISEASE_INFO = {
    'Influenza (Flu)': {
        'doctor': 'General Physician',
        'precautions': ["Rest and sleep", "Drink plenty of fluids", "Take pain relievers if needed", "Avoid contact with others"]
    },
    'Common Cold': {
        'doctor': 'General Physician',
        'precautions': ["Stay hydrated", "Get plenty of rest", "Use saline nasal drops", "Warm gargles"]
    },
    'Food Poisoning': {
        'doctor': 'Gastroenterologist',
        'precautions': ["Drink oral rehydration solutions (ORS)", "Eat bland foods like rice or toast", "Avoid dairy and caffeine", "Rest your stomach"]
    },
    'Covid-19': {
        'doctor': 'Pulmonologist / General Physician',
        'precautions': ["Isolate yourself", "Monitor oxygen levels", "Wear a mask", "Stay hydrated"]
    },
    'Diabetes': {
        'doctor': 'Endocrinologist',
        'precautions': ["Monitor blood sugar levels", "Maintain a healthy diet", "Exercise regularly", "Take prescribed medication"]
    },
    'Hypertension': {
        'doctor': 'Cardiologist',
        'precautions': ["Reduce salt intake", "Exercise regularly", "Manage stress", "Avoid smoking and alcohol"]
    },
    'Migraine': {
        'doctor': 'Neurologist',
        'precautions': ["Rest in a dark, quiet room", "Apply a cold compress to forehead", "Stay hydrated", "Avoid known triggers"]
    },
    'Malaria': {
        'doctor': 'Infectious Disease Specialist / General Physician',
        'precautions': ["Take prescribed antimalarial medication", "Use mosquito nets and repellents", "Stay hydrated", "Monitor temperature"]
    },
    'Dengue': {
        'doctor': 'General Physician',
        'precautions': ["Rest extensively", "Drink plenty of fluids/electrolytes", "Monitor platelet count", "Avoid NSAIDs like ibuprofen (use paracetamol)"]
    },
    'Heart Attack': {
        'doctor': 'Cardiologist',
        'precautions': ["Call emergency services immediately", "Chew an aspirin if recommended by doctor", "Sit down and stay calm", "Loosen tight clothing"]
    },
    'Allergies': {
        'doctor': 'Allergist / Immunologist',
        'precautions': ["Avoid allergen triggers", "Take antihistamines if prescribed", "Keep indoor air clean", "Use a cool compress"]
    },
    'Dermatitis': {
        'doctor': 'Dermatologist',
        'precautions': ["Moisturize skin regularly", "Avoid harsh soaps and chemicals", "Do not scratch the skin", "Apply cool, damp compresses"]
    },
    'Asthma': {
        'doctor': 'Pulmonologist',
        'precautions': ["Use quick-relief inhaler", "Avoid asthma triggers (smoke, dust)", "Monitor breathing", "Stay calm during attacks"]
    },
    'Typhoid': {
        'doctor': 'General Physician',
        'precautions': ["Drink boiled or bottled water", "Eat thoroughly cooked food", "Maintain good hand hygiene", "Complete the full antibiotic course"]
    }
}

def load_model_bundle():
    """Loads the model bundle pickle file containing models, features and encoder."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    return bundle

def get_symptoms_list():
    """Returns the list of symptoms the models expect."""
    try:
        bundle = load_model_bundle()
        return bundle['symptoms']
    except Exception:
        # Fallback list if model not yet trained
        return [
            'fever', 'headache', 'vomiting', 'nausea', 'fatigue', 'cough', 'cold', 
            'joint_pain', 'itching', 'skin_rash', 'chest_pain', 'stomach_pain', 
            'diarrhoea', 'back_pain', 'high_fever', 'breathlessness', 'loss_of_appetite', 
            'sweating', 'dizziness', 'blurred_vision', 'muscle_pain', 'sore_throat', 
            'runny_nose', 'sneezing', 'chills', 'frequent_urination', 'increased_thirst', 
            'weight_loss', 'wheezing', 'cough_with_sputum'
        ]

def get_symptoms_with_counts():
    """Returns the list of symptoms and their frequency counts from the dataset."""
    try:
        bundle = load_model_bundle()
        return bundle['symptoms'], bundle.get('symptom_counts', {})
    except Exception:
        symptoms = get_symptoms_list()
        # Mock counts if training has not been executed yet
        return symptoms, {s: 0 for s in symptoms}

def predict_disease(selected_symptoms, model_type='decision_tree'):
    """
    Predicts the disease based on user symptoms.
    
    Parameters:
    - selected_symptoms: list of symptoms present in the user
    - model_type: 'decision_tree', 'naive_bayes', or 'random_forest'
    
    Returns:
    - dictionary containing predicted disease, confidence, doctor, precautions, and model details.
    """
    bundle = load_model_bundle()
    symptoms = bundle['symptoms']
    label_encoder = bundle['label_encoder']
    
    if model_type not in bundle['models']:
        raise ValueError(f"Invalid model_type. Must be one of {list(bundle['models'].keys())}")
        
    model_info = bundle['models'][model_type]
    clf = model_info['model']
    
    # 1. Create binary feature vector
    input_dict = {sym: 0 for sym in symptoms}
    for sym in selected_symptoms:
        sym_norm = sym.lower().strip().replace(' ', '_')
        if sym_norm in input_dict:
            input_dict[sym_norm] = 1
            
    input_df = pd.DataFrame([input_dict])
    
    # 2. Predict disease label
    predicted_encoded = clf.predict(input_df)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_encoded])[0]
    
    # 3. Calculate confidence score (probability)
    probabilities = clf.predict_proba(input_df)[0]
    confidence = float(probabilities[predicted_encoded]) * 100
    
    # 4. Enrich with suggested doctor and precautions
    info = DISEASE_INFO.get(predicted_disease, {
        'doctor': 'General Physician',
        'precautions': ["Rest and stay hydrated", "Monitor symptoms", "Consult a physician if symptoms worsen"]
    })
    
    return {
        'disease': predicted_disease,
        'confidence': round(confidence, 2),
        'suggested_doctor': info['doctor'],
        'precautions': info['precautions'],
        'model_used': model_type.replace('_', ' ').title(),
        'model_accuracy': model_info['test_acc']
    }
