from fastapi import APIRouter, HTTPException
from app.services.recommendation_service import RecommendationService
router = APIRouter()
recommendation_service = RecommendationService()
@router.get('/api/recommendations/user/{user_id}')
async def get_user_recommendations(user_id: int):
	try:
		recommendations = await recommendation_service.get_recommendations(user_id)
		return recommendations
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/recommendations/similar/{vehicle_id}')
async def get_similar_vehicles(vehicle_id: int):
	try:
		similar = await recommendation_service.get_similar_vehicles(vehicle_id)
		return similar
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
