from typing import Dict, List, Optional, TYPE_CHECKING
from app.schemas.schemas import RecommendationResponse, VehicleRecommendation
from app.exceptions.recommendation_exceptions import InsufficientDataError
from app.repositories.vehicle_repository import VehicleRepository
from app.repositories.user_repository import UserRepository
from app.services.score_combiner import ScoreCombiner
import asyncio
if TYPE_CHECKING:
    from app.interfaces.recommendation_interfaces import (
        IContentBasedRecommender,
        ICollaborativeRecommender,
    )

from app.services.model_serving_service import ModelServingService
class HybridRecommender:
    """
    Hybrid recommendation engine that combines content-based + collaborative filtering.
    """

    def __init__(
        self,
        user_repo: UserRepository,
        vehicle_repo: VehicleRepository,
        content_recommender: "IContentBasedRecommender",
        collab_recommender: "ICollaborativeRecommender",
        score_combiner: ScoreCombiner,
    ):
        self.user_repo = user_repo
        self.vehicle_repo = vehicle_repo
        self.content_recommender = content_recommender
        self.collab_recommender = collab_recommender
        self.score_combiner = score_combiner

    async def get_recommendations(self, user_id: int, top_n: int = 10) -> RecommendationResponse:
        user_interactions = await self.user_repo.get_user_interactions(user_id)

        if not user_interactions or len(user_interactions) <= 10:
            return await self._handle_cold_start(user_id, user_interactions, top_n)

        return await self._compute_hybrid_scores(user_id, user_interactions, top_n, 0.5, 0.5)
    
    async def _handle_cold_start(
        self, user_id: int, user_interactions: Optional[List[dict]], top_n: int
    ) -> RecommendationResponse:
        if not user_interactions:
            raise InsufficientDataError(user_id)

        interaction_count = len(user_interactions)
        if interaction_count <= 3:
            content_weight, collab_weight = 0.9, 0.1
        else:
            content_weight, collab_weight = 0.7, 0.3

        return await self._compute_hybrid_scores(
            user_id, user_interactions, top_n, content_weight, collab_weight
        )

    async def _compute_hybrid_scores(
        self, user_id: int, user_interactions: List[dict], top_n: int,
        content_weight: float, collab_weight: float
    ) -> RecommendationResponse:
        await asyncio.gather(self.content_recommender.model_serving.load_model("user_similarity"), self.collab_recommender.model_serving.load_model("collaborative"))        

        collab_scores: Dict[int, float] = await self.collab_recommender.get_collaborative_recommendations(
            user_id, top_n * 3
        ) or {}

        content_scores: Dict[int, float] = {}
        for inter in user_interactions:
            vid = int(inter.get("vehicle_id"))
            weight = float(inter.get("weight", 1.0))
            if not vid:
                continue
            similar_list = await self.content_recommender.get_similar_vehicles_scores(
                vid, top_n * 3, "user_similarity"
            ) or []
            for sv in similar_list:
                sid = int(sv["vehicle_id"])
                sim = float(sv["similarity_score"])
                content_scores[sid] = content_scores.get(sid, 0.0) + sim * weight

        if content_scores:
            max_c = max(content_scores.values()) or 1.0
            for k in list(content_scores.keys()):
                content_scores[k] = content_scores[k] / max_c

        hybrid_scores = self.score_combiner.combine(content_scores, collab_scores, content_weight, collab_weight)

        ranked = sorted(hybrid_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

        recommendations: List[VehicleRecommendation] = []
        for vid, score in ranked:
            row = self.vehicle_repo.get_vehicle_by_id(vid)
            if not row:
                continue
            features = self.vehicle_repo.extract_vehicle_features(row)
            recommendations.append(
                VehicleRecommendation(vehicle_id=vid, score=float(score), features=features)
            )

        return RecommendationResponse(recommendations=recommendations, model_type="hybrid")
