import logging
from redis import Redis
from app.interfaces.recommendation_interfaces import (
    IRecommendationOrchestrator,
    IContentBasedRecommender,
    ICollaborativeRecommender,
    IScoreCombiner,
)
from app.repositories.vehicle_repository import VehicleRepository
from app.repositories.user_repository import UserRepository
from app.services.caching_service import CachingService
from app.recommenders.content_based_recommender import ContentBasedRecommender
from app.recommenders.collaborative_based_recommender import CollaborativeBasedRecommender
from app.recommenders.hybrid_based_recommender import HybridRecommender
from app.services.score_combiner import ScoreCombiner 
from app.services.model_serving_service import ModelServingService
from app.db import DatabaseManager
from app.services.user_context_service import MLUserContextService
logger = logging.getLogger(__name__)


class DependencyContainer:
    def __init__(
        self,
        orchestrator: IRecommendationOrchestrator,
        vehicle_repo: VehicleRepository,
        user_repo: UserRepository,
        model_serving_service: ModelServingService,
        redis_client: Redis,
        caching_service: CachingService,
        db_manager: DatabaseManager 
    ):
        self._orchestrator = orchestrator
        self._vehicle_repo = vehicle_repo
        self._user_repo = user_repo
        self._model_serving_service = model_serving_service
        self._redis_client = redis_client
        self._caching_service = caching_service
        self._db_manager = db_manager 

        self._instances = {}

        logger.info("DependencyContainer initialized with all services")

    @property
    def orchestrator(self) -> IRecommendationOrchestrator:
        return self._orchestrator

    @property
    def db_manager(self) -> DatabaseManager:
        return self._db_manager
    
    @property
    def vehicle_repo(self) -> VehicleRepository:
        return self._vehicle_repo

    @property
    def user_repo(self) -> UserRepository:
        return self._user_repo

    @property
    def model_serving_service(self) -> ModelServingService:
        return self._model_serving_service

    @property
    def redis_client(self) -> Redis:
        return self._redis_client

    def get(self, interface):
            """Resolve dependencies by interface type."""
            if interface in self._instances:
                return self._instances[interface]

            if interface.__name__ == "IContentBasedRecommender":
                self._instances[interface] = ContentBasedRecommender(
                    vehicle_repository=self._vehicle_repo,
                    model_service=self.model_serving_service,      
                    caching_service=self._caching_service
                )
            elif interface.__name__ == "ICollaborativeRecommender":
                self._instances[interface] = CollaborativeBasedRecommender(
                    model_serving=self.model_serving_service,        
                )
            elif interface.__name__ == "IHybridRecommender":
                self._instances[interface] = HybridRecommender(
                    user_repo=self._user_repo,
                    vehicle_repo=self._vehicle_repo,
                    content_recommender=self.get(IContentBasedRecommender),
                    collab_recommender=self.get(ICollaborativeRecommender),
                    score_combiner=self.get(IScoreCombiner)
                )
            elif interface.__name__ == "IScoreCombiner":
                self._instances[interface] = ScoreCombiner()
            elif interface.__name__ == "IRecommendationOrchestrator":
                self._instances[interface] = self._orchestrator
            elif interface.__name__ == "MLUserContextService":
                self._instances[interface] = MLUserContextService(db=self._db_manager,cache=self._caching_service)
            else:
                raise ValueError(f"No binding found for {interface}")

            return self._instances[interface]