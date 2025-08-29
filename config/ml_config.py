from dataclasses import dataclass, field
from typing import Dict
from config.app_config import settings
import json

@dataclass
class MLConfig:
    svd_components: int = int(settings.SVD_COMPONENTS)
    random_state: int = int(settings.RANDOM_STATE)
    max_iter: int = int(settings.MAX_ITER)

    content_similarity_threshold: float = float(settings.CONTENT_SIMILARITY_THRESHOLD)
    top_k_similar: int = int(settings.TOP_K_SIMILAR)

    hybrid_content_weight: float = float(settings.HYBRID_CONTENT_WEIGHT)
    hybrid_collaborative_weight: float = float(settings.HYBRID_COLLABORATIVE_WEIGHT)

    vehicle_feature_weights: Dict[str, float] = field(
        default_factory=lambda: json.loads(settings.VEHICLE_FEATURE_WEIGHTS)
    )
    user_feature_weights: Dict[str, float] = field(
        default_factory=lambda: json.loads(settings.USER_FEATURE_WEIGHTS)
    )
    interaction_weights: Dict[str, float] = field(
        default_factory=lambda: json.loads(settings.INTERACTION_WEIGHTS)
    )
