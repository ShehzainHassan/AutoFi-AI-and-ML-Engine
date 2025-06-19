from pydantic import BaseModel
from typing import List, Dict

class VehicleRecommendation(BaseModel):
    vehicle_id: int
    score: float
    features: Dict[str, str] 

class RecommendationResponse(BaseModel):
    recommendations: List[VehicleRecommendation]
    model_type: str 

class SimilarVehicle(BaseModel):
    vehicle_id: int
    similarity_score: float
    features: Dict[str, str]

class SimilarVehiclesResponse(BaseModel):
    vehicle_id: int
    similar_vehicles: List[SimilarVehicle]
