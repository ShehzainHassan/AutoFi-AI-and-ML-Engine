from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import recommendation_routes
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting AutoFi Vehicle Recommendation API...")
    try:
        from app.services.recommendation_service import RecommendationService
        recommendation_service = RecommendationService(vehicle_limit=20000)
        recommendation_routes.recommendation_service = recommendation_service
        logger.info("✅ Recommendation service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize recommendation service: {e}")
        logger.warning("⚠️  API will run but recommendations may not work")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down AutoFi Vehicle Recommendation API...")
    if hasattr(recommendation_routes, 'recommendation_service') and recommendation_routes.recommendation_service:
        try:
            recommendation_routes.recommendation_service.close()
            logger.info("✅ Recommendation service closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing recommendation service: {e}")

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
    
