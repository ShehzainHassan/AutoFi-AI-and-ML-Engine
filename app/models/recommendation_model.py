import joblib
import os

MODEL_DIR = "trained_models"

def save_collaborative_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[SAVE] Saving collaborative model...")
    joblib.dump(model[0], f'{MODEL_DIR}/collaborative_model.pkl')
    joblib.dump(model[1], f'{MODEL_DIR}/user_features.npy')
    joblib.dump(model[2], f'{MODEL_DIR}/vehicle_features.npy')
    joblib.dump(model[3], f'{MODEL_DIR}/interaction_matrix.pkl')
    print("[SAVE] Collaborative model saved successfully.")

def load_collaborative_model():
    if not os.path.exists(f'{MODEL_DIR}/collaborative_model.pkl'):
        print("[LOAD] Collaborative model not found.")
        return None
    
    print("[LOAD] Collaborative model found. Loading...")
    model = (
        joblib.load(f'{MODEL_DIR}/collaborative_model.pkl'),
        joblib.load(f'{MODEL_DIR}/user_features.npy'),
        joblib.load(f'{MODEL_DIR}/vehicle_features.npy'),
        joblib.load(f'{MODEL_DIR}/interaction_matrix.pkl'),
    )
    print("[LOAD] Collaborative model loaded successfully.")
    return model

def save_content_model(similarity_topk):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[SAVE] Saving content-based model (vehicle TOP-K)...")
    joblib.dump(similarity_topk, f'{MODEL_DIR}/similarity_topk_vehicle.pkl')
    print("[SAVE] Content-based model (vehicle) saved successfully.")

def load_content_model():
    file_path = f'{MODEL_DIR}/similarity_topk_vehicle.pkl'
    if not os.path.exists(file_path):
        print("[LOAD] Content-based model (vehicle) not found.")
        return None
    
    print("[LOAD] Content-based model (vehicle) found. Loading...")
    similarity_topk = joblib.load(file_path)
    print("[LOAD] Content-based model (vehicle) loaded successfully.")
    return similarity_topk

def save_user_content_model(similarity_topk):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[SAVE] Saving content-based model (user TOP-K)...")
    joblib.dump(similarity_topk, f'{MODEL_DIR}/similarity_topk_user.pkl')
    print("[SAVE] Content-based model (user) saved successfully.")

def load_user_content_model():
    file_path = f'{MODEL_DIR}/similarity_topk_user.pkl'
    if not os.path.exists(file_path):
        print("[LOAD] Content-based model (user) not found.")
        return None
    
    print("[LOAD] Content-based model (user) found. Loading...")
    similarity_topk = joblib.load(file_path)
    print("[LOAD] Content-based model (user) loaded successfully.")
    return similarity_topk
