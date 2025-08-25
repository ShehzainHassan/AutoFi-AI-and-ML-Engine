import numpy as np
from app.exceptions.recommendation_exceptions import (
    UserNotFoundError,
    ModelNotAvailableError,
)
from app.services.model_serving_service import ModelServingService

class CollaborativeBasedRecommender:
    def __init__(self, model_serving: ModelServingService):
        self.model_serving = model_serving

    async def get_collaborative_recommendations(self, user_id: int, top_n: int):
        collaborative_model = await self.model_serving.load_model("collaborative")
        if collaborative_model is None:
            raise ModelNotAvailableError("Collaborative model is loading, try again later")

        interaction_matrix = collaborative_model.get("interaction_matrix")
        user_features = collaborative_model.get("user_features")
        vehicle_features = collaborative_model.get("vehicle_features")

        if (
        interaction_matrix is None
            or user_features is None
            or vehicle_features is None
        ):
            raise ModelNotAvailableError("Collaborative model is not available or corrupted")

        if user_id not in interaction_matrix.index:
            raise UserNotFoundError(user_id)

        user_index = interaction_matrix.index.get_loc(user_id)
        user_vector = user_features[user_index]

        scores = np.dot(vehicle_features, user_vector)
        vehicle_ids = interaction_matrix.columns.values

        min_score, max_score = scores.min(), scores.max()
        norm_scores = (scores - min_score) / ((max_score - min_score) or 1.0)

        result = {
            int(v_id): float(norm_scores[i]) for i, v_id in enumerate(vehicle_ids)
        }
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
