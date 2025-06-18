from fastapi import APIRouter, HTTPException
from app.services.recommendation_service import RecommendationService

router = APIRouter()
recommendation_service = RecommendationService()

@router.get('/api/recommendations/user/{user_id}')
async def get_user_recommendations(user_id: int):
    """
    Returns top N hybrid recommendations for a given user_id
    """
    try:
        # Trigger full model training pipeline if needed
        collaborative_model = recommendation_service.train_models()
        models = (recommendation_service, collaborative_model, collaborative_model[3])

        recommendations = recommendation_service.get_hybrid_recommendations(user_id, models, top_n=5)
        return {"user_id": user_id, "recommended_vehicle_ids": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/recommendations/similar/{vehicle_id}')
async def get_similar_vehicles(vehicle_id: int, top_n: int = 5):
    """
    Returns top N similar vehicles based on content-based model
    """
    try:
        if recommendation_service.similarity_matrix is None or recommendation_service.vehicle_ids is None:
            print("Training content-based model (from API)...")
            _, features_df = recommendation_service.prepare_data_for_ml()
            recommendation_service.train_content_based_model(features_df)

        similar_ids = recommendation_service.get_similar_vehicles_content_based(vehicle_id, n=top_n)
        return {"vehicle_id": vehicle_id, "similar_vehicle_ids": similar_ids}
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