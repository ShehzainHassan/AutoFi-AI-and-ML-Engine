import logging
from typing import Optional

from app.interfaces.recommendation_interfaces import (
    IVehicleRepository,
    IMLModelService,
    IRecommendationOrchestrator,
    IUserRepository,
    IContentBasedRecommender,
)
from app.schemas import schemas
from app.exceptions.recommendation_exceptions import (
    UserNotFoundError,
    VehicleNotFoundError,
)
from app.services.caching_service import CachingService
from app.strategies.recommendation_strategies import RecommendationStrategy, RecommendationStrategyFactory
class RecommendationOrchestrator(IRecommendationOrchestrator):
    """
    High-level orchestration: delegates hybrid user recommendations to the async, cached path.
    Keeps 'similar vehicles' in-memory path for low-latency content-based lookups.
    """

    def __init__(
        self,
        vehicle_repository: IVehicleRepository,
        user_repository: IUserRepository,
        caching_service: CachingService,
        strategy_factory: RecommendationStrategyFactory,
        ml_service: IMLModelService,
        logger: logging.Logger,
        default_strategy: RecommendationStrategy = RecommendationStrategy.HYBRID,
    ):
        self.vehicle_repository = vehicle_repository
        self.user_repository = user_repository
        self.cache = caching_service
        self.strategy_factory = strategy_factory
        self.ml_service = ml_service
        self.logger = logger
        self.default_strategy = default_strategy

    async def get_recommendations(self, user_id: int, top_n: int, strategy: Optional[RecommendationStrategy] = None) -> schemas.RecommendationResponse:
        """
        Fetch recommendations for a user, using the configured strategy (default = hybrid).
        """
        strategy = strategy or self.default_strategy
        self.logger.info(f"Fetching {strategy.value} recommendations for user_id={user_id}")

        if not await self.user_repository.user_exists(user_id):
            raise UserNotFoundError(user_id)

        recommender = self.strategy_factory.create_recommender(strategy)
        return await recommender.get_recommendations(user_id=user_id, top_n=top_n)

    async def get_similar_vehicles(self, vehicle_id: int, top_n: int):
        row = self.vehicle_repository.get_vehicle_by_id(vehicle_id)
        if not row:
            raise VehicleNotFoundError(vehicle_id)

        content_recommender: IContentBasedRecommender = self.strategy_factory.create_recommender(RecommendationStrategy.CONTENT_BASED)
        return await content_recommender.get_similar_vehicles(vehicle_id, top_n)

    async def train_all_models(self):
        await self.ml_service.train_vehicle_similarity_model()
        await self.ml_service.train_user_similarity_model()
        await self.ml_service.train_collaborative_model()