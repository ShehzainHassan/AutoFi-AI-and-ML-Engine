from typing import List, Dict, TYPE_CHECKING
from app.schemas import schemas
from app.repositories.vehicle_repository import VehicleRepository
from app.services.caching_service import CachingService
from app.services.model_serving_service import ModelServingService
from app.exceptions.recommendation_exceptions import VehicleNotFoundError, ModelNotAvailableError

class ContentBasedRecommender: 
    """
    Handles content-based recommendation logic only.
    Loads similarity models via MLModelService, applies caching, enriches results with vehicle features.
    """

    def __init__(self, vehicle_repository: VehicleRepository, model_service: ModelServingService, caching_service: CachingService):
        self.vehicle_repository = vehicle_repository
        self.model_serving = model_service
        self.cache = caching_service

    async def get_similar_vehicles(self, vehicle_id: int, top_n: int) -> schemas.SimilarVehiclesResponse:
        # Check cache first
        cached = await self.cache.get_cached_vehicle_similarity(vehicle_id, top_n)
        if cached:
            similar_raw = cached
        else:
            similar_raw = await self._compute_similar_vehicles(vehicle_id, top_n)
            await self.cache.set_cached_vehicle_similarity(vehicle_id, top_n, similar_raw)

        similar: List[schemas.SimilarVehicle] = []
        for sv in similar_raw:
            srow = self.vehicle_repository.get_vehicle_by_id(sv["vehicle_id"])
            if not srow:
                continue
            features = self.vehicle_repository.extract_vehicle_features(srow)
            similar.append(
                schemas.SimilarVehicle(
                    vehicle_id=sv["vehicle_id"],
                    similarity_score=sv["similarity_score"],
                    features=features,
                )
            )

        return schemas.SimilarVehiclesResponse(
            vehicle_id=vehicle_id,
            similar_vehicles=similar,
            source="content-based",
        )

    async def _compute_similar_vehicles(self, vehicle_id: int, top_n: int, model_name="vehicle_similarity") -> List[Dict]:
        model = await self.model_serving.load_model(model_name)
        if model is None:
            raise ModelNotAvailableError("content-based model not available")
        if vehicle_id not in model:
            raise VehicleNotFoundError(vehicle_id)

        sims = model[vehicle_id][:top_n]
        return [{"vehicle_id": vid, "similarity_score": score} for vid, score in sims]

    async def get_similar_vehicles_scores(self, vehicle_id: int, top_n: int, model_name="user_similarity") -> List[Dict]:
        model = await self.model_serving.load_model(model_name)
        
        if model is None:
            raise ModelNotAvailableError("content-based model not available")

        cached = await self.cache.get_cached_vehicle_similarity(vehicle_id, top_n)
        if cached:
            return cached

        similar_raw = await self._compute_similar_vehicles(
            vehicle_id, top_n, model_name=model_name
        )
        await self.cache.set_cached_vehicle_similarity(vehicle_id, top_n, similar_raw)
        return similar_raw
