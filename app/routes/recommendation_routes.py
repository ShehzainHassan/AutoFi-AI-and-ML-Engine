from fastapi import APIRouter, HTTPException
from app.services.recommendation_service import RecommendationService
from app.schemas.schemas import RecommendationResponse, SimilarVehiclesResponse

router = APIRouter()
recommendation_service = RecommendationService()

@router.get('/api/recommendations/user/{user_id}', response_model=RecommendationResponse)
async def get_recommendations(user_id: int):
    """
    Returns top N hybrid recommendations for a given user_id
    """
    try:
        if recommendation_service.collaborative_model is None:
            recommendation_service.train_all_models()
            recommendation_service.save_models()

        recommendations = recommendation_service.get_hybrid_recommendations(user_id, top_n=5)
        return recommendations

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/recommendations/similar/{vehicle_id}', response_model=SimilarVehiclesResponse)
async def get_similar_vehicles(vehicle_id: int, top_n: int = 5):
    """
    Returns top N similar vehicles based on content-based model
    """
    try:
        if recommendation_service.vehicle_similarity_matrix is None or recommendation_service.vehicle_ids is None:
            recommendation_service.train_vehicle_similarity_model()
            recommendation_service.save_models()

        similar_vehicles = recommendation_service.get_similar_vehicles(vehicle_id, top_n)

        return SimilarVehiclesResponse(vehicle_id=vehicle_id,similar_vehicles=similar_vehicles)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/recommendations/interactions-summary')
def get_interactions_summary():
    try:
        df = recommendation_service.load_interactions_summary()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/recommendations/vehicle-features')
def get_vehicle_features():
    try:
        df = recommendation_service.load_vehicle_features()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
