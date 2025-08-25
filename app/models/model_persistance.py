import joblib
import os
from typing import Dict, List, Tuple, Optional

MODEL_DIR = "trained_models"

def save_collaborative_model(model: dict) -> None:
    """
    Persist the collaborative model as a single dict (svd, matrices).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "collaborative_model.pkl")
    print("[SAVE] Saving collaborative model ...")
    joblib.dump(model, path)
    print("[SAVE] Collaborative model saved.")

def load_collaborative_model() -> Optional[dict]:
    path = os.path.join(MODEL_DIR, "collaborative_model.pkl")
    if not os.path.exists(path):
        print("[LOAD] Collaborative model not found.")
        return None
    print("[LOAD] Collaborative model found. Loading ...")
    model = joblib.load(path)
    print("[LOAD] Collaborative model loaded.")
    return model

def save_content_model(similarity_topk: Dict[int, List[Tuple[int, float]]]) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "similarity_topk_vehicle.pkl")
    print("[SAVE] Saving content-based model (vehicle TOP-K) ...")
    joblib.dump(similarity_topk, path)
    print("[SAVE] Content-based model (vehicle) saved.")

def load_content_model() -> Optional[Dict[int, List[Tuple[int, float]]]]:
    path = os.path.join(MODEL_DIR, "similarity_topk_vehicle.pkl")
    if not os.path.exists(path):
        print("[LOAD] Content-based model (vehicle) not found.")
        return None
    print("[LOADING] Content-based model (vehicle)...")
    sim = joblib.load(path)
    print("[LOADED] Content-based model (vehicle) loaded.")
    return sim

def save_user_content_model(similarity_topk: Dict[int, List[Tuple[int, float]]]) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "similarity_topk_user.pkl")
    print("[SAVE] Saving content-based model (user TOP-K) ...")
    joblib.dump(similarity_topk, path)
    print("[SAVE] Content-based model (user) saved.")

def load_user_content_model() -> Optional[Dict[int, List[Tuple[int, float]]]]:
    path = os.path.join(MODEL_DIR, "similarity_topk_user.pkl")
    if not os.path.exists(path):
        print("[LOAD] Content-based model (user) not found.")
        return None
    print("[LOAD] Content-based model (user) found. Loading ...")
    sim = joblib.load(path)
    print("[LOAD] Content-based model (user) loaded.")
    return sim
