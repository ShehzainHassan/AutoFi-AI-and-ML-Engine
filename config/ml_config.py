from dataclasses import dataclass, field
from typing import Dict
from config.app_config import settings
import json


DEFAULT_VEHICLE_FEATURE_WEIGHTS = {
    "Horsepower": 5.0, "TorqueFtLbs": 5.0, "EngineSize": 4.5, "ZeroTo60MPH": 4.5,
    "DrivetrainType": 4.0, "CO2Emissions": 3.5, "Transmission": 3.5, "Price": 3.0,
    "Model": 2.5, "Make": 2.0, "Year": 1.5, "Color": 1.5, "FuelType": 1.0,
    "CityMPG": 0.8, "Mileage": 0.5, "Status": 0.5
}

DEFAULT_USER_FEATURE_WEIGHTS = {
    "Price": 5.0, "FuelType": 4.5, "DrivetrainType": 4.25, "CO2Emissions": 4.0,
    "CityMPG": 4.0, "Horsepower": 3.5, "TorqueFtLbs": 3.5, "EngineSize": 3.0,
    "Color": 2.5, "Transmission": 2.5, "Mileage": 2.0, "ZeroTo60MPH": 1.5,
    "Status": 1.5, "Model": 1.0, "Make": 1.0, "Year": 0.8
}

DEFAULT_INTERACTION_WEIGHTS = {
    "favorite-added": 5.0, "contacted-seller": 4.0, "share": 3.0, "view": 1.0
}


def load_weights(config_value: str, fallback: Dict[str, float]) -> Dict[str, float]:
    """Safely load JSON weights from config, falling back to defaults on error."""
    try:
        parsed = json.loads(config_value) if config_value else {}
        if isinstance(parsed, dict):
            return {**fallback, **parsed}
    except Exception:
        pass
    return fallback


def validate_weights(weights: Dict[str, float], name: str) -> Dict[str, float]:
    """Validate that weights are non-negative floats."""
    for key, value in weights.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} weight '{key}' must be a number, got {type(value)}")
        if value < 0:
            raise ValueError(f"{name} weight '{key}' cannot be negative (got {value})")
    return weights


@dataclass
class MLConfig:
    svd_components: int = int(getattr(settings, "SVD_COMPONENTS", 50))
    random_state: int = int(getattr(settings, "RANDOM_STATE", 42))
    max_iter: int = int(getattr(settings, "MAX_ITER", 300))

    content_similarity_threshold: float = float(getattr(settings, "CONTENT_SIMILARITY_THRESHOLD", 0.1))
    top_k_similar: int = int(getattr(settings, "TOP_K_SIMILAR", 200))

    hybrid_content_weight: float = float(getattr(settings, "HYBRID_CONTENT_WEIGHT", 0.5))
    hybrid_collaborative_weight: float = float(getattr(settings, "HYBRID_COLLABORATIVE_WEIGHT", 0.5))

    vehicle_feature_weights: Dict[str, float] = field(
        default_factory=lambda: validate_weights(
            load_weights(getattr(settings, "VEHICLE_FEATURE_WEIGHTS", None), DEFAULT_VEHICLE_FEATURE_WEIGHTS),
            "Vehicle"
        )
    )
    user_feature_weights: Dict[str, float] = field(
        default_factory=lambda: validate_weights(
            load_weights(getattr(settings, "USER_FEATURE_WEIGHTS", None), DEFAULT_USER_FEATURE_WEIGHTS),
            "User"
        )
    )
    interaction_weights: Dict[str, float] = field(
        default_factory=lambda: validate_weights(
            load_weights(getattr(settings, "INTERACTION_WEIGHTS", None), DEFAULT_INTERACTION_WEIGHTS),
            "Interaction"
        )
    )
