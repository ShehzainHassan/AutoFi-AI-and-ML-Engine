from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

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
    source: str

class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str
    timestamp: datetime

