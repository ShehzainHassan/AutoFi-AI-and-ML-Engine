# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio
import time
import os
from redis.asyncio import Redis
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.db import DatabaseManager
from app.repositories.vehicle_repository import VehicleRepository
from app.repositories.user_repository import UserRepository
from app.services.ml_service import MLModelService
from app.services.recommendation_orchestrator import RecommendationOrchestrator
from app.services.caching_service import CachingService
from app.middleware.rate_limit_middleware import limiter
from app.dependencies.dependency_container import DependencyContainer
from config.ml_config import MLConfig
from config.app_config import settings
from .routes import recommendation_routes
from app.services.model_serving_service import ModelServingService
from app.strategies.recommendation_strategies import RecommendationStrategyFactory, RecommendationStrategy

APP_VERSION = "1.0.0"
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_manager = DatabaseManager()

container: DependencyContainer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global container
    start_time = time.time()
    logger.info("Starting AutoFi Vehicle Recommendation API...")

    async def retry_async(coro_factory, name: str):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await coro_factory()
            except Exception as e:
                logger.warning(f"[{name}] Retry {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error(f"[{name}] All retries failed.")
                    raise

    async def init_recommendation_system():
        try:
            # DB
            await retry_async(db_manager.initialize, "Database Init")
            pool = db_manager.pool

            # Redis
            redis_client = Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            await retry_async(redis_client.ping, "Redis Ping")

            # Repositories
            vehicle_repo = VehicleRepository(pool=pool, vehicle_limit=20000)
            user_repo = UserRepository(pool=pool)

            # Load Vehicle Features in background
            asyncio.create_task(vehicle_repo.load_vehicle_features())

            # Services
            ml_config = MLConfig()
            model_serving = ModelServingService(max_workers=4)
            caching_service = CachingService(redis_client=redis_client)

            ml_service = MLModelService(
                user_repo=user_repo,
                vehicle_repo=vehicle_repo,
                model_serving=model_serving,
                config=ml_config
            )

            strategy_factory = RecommendationStrategyFactory(None)  

            orchestrator = RecommendationOrchestrator(
                vehicle_repository=vehicle_repo,
                user_repository=user_repo,
                caching_service=caching_service,
                strategy_factory=strategy_factory,
                ml_service=ml_service,
                logger=logger,
                default_strategy=RecommendationStrategy.HYBRID
            )

            container = DependencyContainer(
                orchestrator=orchestrator,
                vehicle_repo=vehicle_repo,
                user_repo=user_repo,
                model_serving_service=model_serving,
                redis_client=redis_client,
                caching_service=caching_service,
            )

            strategy_factory.container = container

            app.state.container = container
            logger.info("Recommendation orchestrator initialized successfully")

            # Train models if missing
            def model_exists(model_name: str) -> bool:
                model_paths = {
                    "collaborative": "trained_models/collaborative_model.pkl",
                    "vehicle_similarity": "trained_models/similarity_topk_vehicle.pkl",
                    "user_similarity": "trained_models/similarity_topk_user.pkl",
                }
                path = model_paths.get(model_name)
                return path and os.path.exists(path)

            if not model_exists("vehicle_similarity"):
                logger.info("Vehicle similarity model not found. Training...")
                await ml_service.train_vehicle_similarity_model()

            if not model_exists("user_similarity"):
                logger.info("User similarity model not found. Training...")
                await ml_service.train_user_similarity_model()

            if not model_exists("collaborative"):
                logger.info("Collaborative model not found. Training...")
                await ml_service.train_collaborative_model()

            logger.info("✅ Model training checks completed")

        except Exception as e:
            logger.exception(f"❌ Failed to initialize recommendation system: {e}")
            logger.warning("API will run but recommendations may not work")

    asyncio.create_task(init_recommendation_system())

    yield

    logger.info("Shutting down AutoFi Vehicle Recommendation API...")
    try:
        await db_manager.close()
        logger.info("Database pool closed successfully")
    except Exception as e:
        logger.error(f"Error shutting down recommendation system: {e}")

    duration = time.time() - start_time
    logger.info(f"⏱️ Startup duration: {duration:.2f} seconds")


# Create FastAPI app with lifespan hook
app = FastAPI(title="Vehicle Recommendation API", version=APP_VERSION, lifespan=lifespan)
app.state.limiter = limiter

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)

# Rate limit handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routes
app.include_router(recommendation_routes.router, prefix="/api/recommendations", dependencies=[])


@app.get("/")
async def root():
    return {"message": "Vehicle Recommendation API is running.", "version": APP_VERSION}

@app.get("/health")
async def health():
    db_ready = db_manager.pool is not None

    orchestrator_ready = False
    ml_ready = False

    if app.state.container:
        try:
            orchestrator = app.state.container.orchestrator
            orchestrator_ready = orchestrator is not None
        except Exception:
            pass

        try:
            ml_service = app.state.container.ml_service
            ml_ready = getattr(ml_service, "models_loaded", False)
        except Exception:
            pass

    return {
        "db": db_ready,
        "ml_models_loaded": ml_ready,
        "orchestrator_ready": orchestrator_ready,
        "version": APP_VERSION,
    }
