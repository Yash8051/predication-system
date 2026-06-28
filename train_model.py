import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'disease_symptom_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Define all unique symptoms
SYMPTOMS = [
    'fever', 'headache', 'vomiting', 'nausea', 'fatigue', 'cough', 'cold', 
    'joint_pain', 'itching', 'skin_rash', 'chest_pain', 'stomach_pain', 
    'diarrhoea', 'back_pain', 'high_fever', 'breathlessness', 'loss_of_appetite', 
    'sweating', 'dizziness', 'blurred_vision', 'muscle_pain', 'sore_throat', 
    'runny_nose', 'sneezing', 'chills', 'frequent_urination', 'increased_thirst', 
    'weight_loss', 'wheezing', 'cough_with_sputum'
]

# Define disease symptom mapping profiles
DISEASE_PROFILES = {
    'Influenza (Flu)': {
        'primary': ['fever', 'cough', 'fatigue', 'headache', 'muscle_pain', 'sore_throat', 'runny_nose'],
        'secondary': ['chills', 'sweating', 'loss_of_appetite']
    },
    'Common Cold': {
        'primary': ['cold', 'cough', 'runny_nose', 'sneezing', 'sore_throat'],
        'secondary': ['fatigue', 'headache']
    },
    'Food Poisoning': {
        'primary': ['vomiting', 'diarrhoea', 'stomach_pain', 'nausea'],
        'secondary': ['fever', 'fatigue', 'loss_of_appetite']
    },
    'Covid-19': {
        'primary': ['fever', 'cough', 'breathlessness', 'fatigue', 'loss_of_appetite'],
        'secondary': ['headache', 'sore_throat', 'muscle_pain', 'chills']
    },
    'Diabetes': {
        'primary': ['increased_thirst', 'frequent_urination', 'fatigue', 'weight_loss'],
        'secondary': ['blurred_vision', 'dizziness']
    },
    'Hypertension': {
        'primary': ['headache', 'dizziness', 'blurred_vision', 'chest_pain'],
        'secondary': ['fatigue']
    },
    'Migraine': {
        'primary': ['headache', 'nausea', 'dizziness'],
        'secondary': ['vomiting', 'blurred_vision']
    },
    'Malaria': {
        'primary': ['high_fever', 'chills', 'sweating', 'headache'],
        'secondary': ['vomiting', 'muscle_pain', 'fatigue']
    },
    'Dengue': {
        'primary': ['high_fever', 'headache', 'joint_pain', 'muscle_pain', 'skin_rash'],
        'secondary': ['nausea', 'vomiting', 'itching']
    },
    'Heart Attack': {
        'primary': ['chest_pain', 'breathlessness', 'sweating'],
        'secondary': ['nausea', 'dizziness']
    },
    'Allergies': {
        'primary': ['sneezing', 'itching', 'runny_nose', 'skin_rash'],
        'secondary': ['cough', 'fatigue']
    },
    'Dermatitis': {
        'primary': ['itching', 'skin_rash'],
        'secondary': ['fatigue']
    },
    'Asthma': {
        'primary': ['breathlessness', 'wheezing', 'cough'],
        'secondary': ['chest_pain']
    },
    'Typhoid': {
        'primary': ['high_fever', 'stomach_pain', 'headache', 'fatigue', 'loss_of_appetite'],
        'secondary': ['vomiting', 'diarrhoea']
    }
}

def generate_dataset(num_samples_per_disease=40):
    """Generates a synthetic disease symptom dataset with noise."""
    np.random.seed(42)
    data = []
    
    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(num_samples_per_disease):
            row = {'Disease': disease}
            for symptom in SYMPTOMS:
                if symptom in profile['primary']:
                    # Primary symptoms are highly likely to be present (85-98%)
                    row[symptom] = 1 if np.random.rand() < 0.90 else 0
                elif symptom in profile['secondary']:
                    # Secondary symptoms are moderately likely to be present (40-70%)
                    row[symptom] = 1 if np.random.rand() < 0.55 else 0
                else:
                    # Other symptoms are very unlikely to be present (2-8% noise)
                    row[symptom] = 1 if np.random.rand() < 0.04 else 0
            data.append(row)
            
    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset generated with {len(df)} samples and saved to {DATASET_PATH}")
    return df

def train_and_save_models():
    """Loads dataset, trains multiple models, and pickles the results."""
    if not os.path.exists(DATASET_PATH):
        df = generate_dataset()
    else:
        df = pd.read_csv(DATASET_PATH)
        
    X = df[SYMPTOMS]
    y = df['Disease']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Model 1: Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_train_acc = dt_model.score(X_train, y_train)
    dt_test_acc = dt_model.score(X_test, y_test)
    
    # Model 2: Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_train_acc = nb_model.score(X_train, y_train)
    nb_test_acc = nb_model.score(X_test, y_test)
    
    # Model 3: Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_acc = rf_model.score(X_train, y_train)
    rf_test_acc = rf_model.score(X_test, y_test)
    
    # Calculate symptom frequencies in dataset
    symptom_counts = {sym: int(df[sym].sum()) for sym in SYMPTOMS}
    
    # Save everything in a model bundle
    model_bundle = {
        'symptoms': SYMPTOMS,
        'symptom_counts': symptom_counts,
        'label_encoder': label_encoder,
        'models': {
            'decision_tree': {
                'model': dt_model,
                'train_acc': round(dt_train_acc * 100, 2),
                'test_acc': round(dt_test_acc * 100, 2)
            },
            'naive_bayes': {
                'model': nb_model,
                'train_acc': round(nb_train_acc * 100, 2),
                'test_acc': round(nb_test_acc * 100, 2)
            },
            'random_forest': {
                'model': rf_model,
                'train_acc': round(rf_train_acc * 100, 2),
                'test_acc': round(rf_test_acc * 100, 2)
            }
        }
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_bundle, f)
        
    print("Models trained and successfully saved to model.pkl")
    print(f"Decision Tree Accuracy  - Train: {dt_train_acc:.2%}, Test: {dt_test_acc:.2%}")
    print(f"Naive Bayes Accuracy    - Train: {nb_train_acc:.2%}, Test: {nb_test_acc:.2%}")
    print(f"Random Forest Accuracy  - Train: {rf_train_acc:.2%}, Test: {rf_test_acc:.2%}")
    
    return model_bundle

if __name__ == '__main__':
    train_and_save_models()
