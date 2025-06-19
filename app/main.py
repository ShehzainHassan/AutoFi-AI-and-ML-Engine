from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.services.recommendation_service import RecommendationService
from .routes import recommendation_routes

recommendation_service = RecommendationService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models")
    recommendation_service.load_models()

    print("Models loaded.")
    yield


app = FastAPI(title="Vehicle Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendation_routes.router)

@app.get("/")
async def root():
    return {"message": "Vehicle Recommendation API is running."}
