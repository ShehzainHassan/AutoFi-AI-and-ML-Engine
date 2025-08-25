from dataclasses import dataclass, field
from typing import Dict

@dataclass
class MLConfig:
    svd_components: int = 50
    random_state: int = 42
    max_iter: int = 300

    content_similarity_threshold: float = 0.1
    top_k_similar: int = 200

    hybrid_content_weight: float = 0.5
    hybrid_collaborative_weight: float = 0.5

    vehicle_feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "Horsepower": 5.0,
        "TorqueFtLbs": 5.0,
        "EngineSize": 4.5,
        "ZeroTo60MPH": 4.5,
        "DrivetrainType": 4.0,
        "CO2Emissions": 3.5,
        "Transmission": 3.5,
        "Price": 3.0,
        "Model": 2.5,
        "Make": 2.0,
        "Year": 1.5,
        "Color": 1.5,
        "FuelType": 1.0,
        "CityMPG": 0.8,
        "Mileage": 0.5,
        "Status": 0.5,
    })

    user_feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "Price": 5.0, "FuelType": 4.5, "DrivetrainType": 4.25, "CO2Emissions": 4.0,
        "CityMPG": 4.0, "Horsepower": 3.5, "TorqueFtLbs": 3.5, "EngineSize": 3.0,
        "Color": 2.5, "Transmission": 2.5, "Mileage": 2.0, "ZeroTo60MPH": 1.5,
        "Status": 1.5, "Model": 1.0, "Make": 1.0, "Year": 0.8
    })

    interaction_weights: Dict[str, float] = field(default_factory=lambda: {
        "favorite-added": 5.0,
        "contacted-seller": 4.0,
        "share": 3.0,
        "view": 1.0,
    })
