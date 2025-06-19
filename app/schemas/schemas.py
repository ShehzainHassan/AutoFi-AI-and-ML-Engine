from pydantic import BaseModel
from typing import List

class UserRecommendationsResponse(BaseModel):
    user_id: int
    recommended_vehicle_ids: List[int]

class SimilarVehiclesResponse(BaseModel):
    vehicle_id: int
    similar_vehicle_ids: List[int]
