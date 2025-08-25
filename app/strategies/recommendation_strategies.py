from enum import Enum
from app.recommenders.hybrid_based_recommender import HybridRecommender

from app.interfaces.recommendation_interfaces import (
    IContentBasedRecommender,
    ICollaborativeRecommender,
    IHybridRecommender,
    IScoreCombiner,
)
from app.dependencies.dependency_container import DependencyContainer


class RecommendationStrategy(Enum):
    CONTENT_BASED = "content_based"
    COLLABORATIVE = "collaborative"
    HYBRID = "hybrid"


class RecommendationStrategyFactory:
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.strategies = {
            RecommendationStrategy.CONTENT_BASED: self._create_content_based,
            RecommendationStrategy.COLLABORATIVE: self._create_collaborative,
            RecommendationStrategy.HYBRID: self._create_hybrid,
        }

    def create_recommender(self, strategy: RecommendationStrategy):
        if strategy not in self.strategies:
            raise ValueError(f"Unknown recommendation strategy: {strategy}")
        return self.strategies[strategy]()

    def _create_content_based(self) -> IContentBasedRecommender:
        return self.container.get(IContentBasedRecommender)

    def _create_collaborative(self) -> ICollaborativeRecommender:
        return self.container.get(ICollaborativeRecommender)

    def _create_hybrid(self) -> IHybridRecommender:
        return HybridRecommender(
            user_repo=self.container.user_repo,
            vehicle_repo=self.container.vehicle_repo,
            content_recommender=self.container.get(IContentBasedRecommender),
            collab_recommender=self.container.get(ICollaborativeRecommender),
            score_combiner=self.container.get(IScoreCombiner),
        )
