from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import recommendation_routes
from app.services.recommendation_service import RecommendationService

recommendation_service = RecommendationService(vehicle_limit=20000)

app = FastAPI(title="Vehicle Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommendation_routes.recommendation_service = recommendation_service
app.include_router(recommendation_routes.router)

@app.get("/")
async def root():
    return {"message": "Vehicle Recommendation API is running."}
    
