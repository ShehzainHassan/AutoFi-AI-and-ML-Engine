import joblib
import os

MODEL_DIR = "trained_models"

def save_collaborative_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model[0], f'{MODEL_DIR}/collaborative_model.pkl')
    joblib.dump(model[1], f'{MODEL_DIR}/user_features.npy')
    joblib.dump(model[2], f'{MODEL_DIR}/vehicle_features.npy')
    joblib.dump(model[3], f'{MODEL_DIR}/interaction_matrix.pkl')

def load_collaborative_model():
    if not os.path.exists(f'{MODEL_DIR}/collaborative_model.pkl'):
        return None
    return (
        joblib.load(f'{MODEL_DIR}/collaborative_model.pkl'),
        joblib.load(f'{MODEL_DIR}/user_features.npy'),
        joblib.load(f'{MODEL_DIR}/vehicle_features.npy'),
        joblib.load(f'{MODEL_DIR}/interaction_matrix.pkl'),
    )

def save_content_model(similarity_matrix, vehicle_ids):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(similarity_matrix, f'{MODEL_DIR}/similarity_matrix.npy')
    joblib.dump(vehicle_ids, f'{MODEL_DIR}/vehicle_ids.npy')

def load_content_model():
    if not os.path.exists(f'{MODEL_DIR}/similarity_matrix.npy'):
        return None, None
    similarity_matrix = joblib.load(f'{MODEL_DIR}/similarity_matrix.npy')
    vehicle_ids = joblib.load(f'{MODEL_DIR}/vehicle_ids.npy')
    return similarity_matrix, vehicle_ids
