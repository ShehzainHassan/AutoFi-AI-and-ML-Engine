from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
from app.schemas.schemas import RecommendationResponse, SimilarVehiclesResponse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.strategies.recommendation_strategies import RecommendationStrategy

class IVehicleRepository(ABC):
    """Database access and in-memory caching for vehicles & interactions."""
    @abstractmethod
    async def get_vehicle_features(self, vehicle_ids: Optional[List[int]] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    async def load_vehicle_features(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_vehicle_by_id(self, vehicle_id: int) -> Optional[Dict[str, Any]]:
        pass

    @staticmethod
    @abstractmethod
    def extract_vehicle_features(row: pd.Series | Dict[str, Any]) -> Dict[str, str]:
        pass

class IMLModelService(ABC):
    """ML model inference, training, and persistence (content-based & collaborative)."""
    @abstractmethod
    async def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    async def train_vehicle_similarity_model(self) -> None:
        pass

    @abstractmethod
    async def train_user_similarity_model(self) -> None:
        pass

    @abstractmethod
    async def train_collaborative_model(self) -> None:
        pass

class IRecommendationOrchestrator(ABC):
    """High-level orchestration of hybrid recommendations."""

    @abstractmethod
    async def get_recommendations(self, user_id: int, top_n: int, strategy: Optional["RecommendationStrategy"] = None) -> RecommendationResponse:
        pass

    @abstractmethod
    async def get_similar_vehicles(self, vehicle_id: int, top_n: int) -> SimilarVehiclesResponse:
        pass

class IUserRepository(ABC):
    """Database access and in-memory caching for users."""
    @abstractmethod
    async def user_exists(self, user_id: int) -> bool:
        pass

    @abstractmethod
    async def load_interactions_summary(self) -> pd.DataFrame:
        pass

    @abstractmethod
    async def get_user_interactions(self, user_id: int) -> List[Dict[str, float]]:
        pass

class IContentBasedRecommender(ABC):
    @abstractmethod
    async def get_similar_vehicles(
        self, vehicle_id: int, top_n: int
    ) -> SimilarVehiclesResponse:
        """Return vehicles similar to the given one (with enrichment)."""
        pass

    @abstractmethod
    async def get_similar_vehicles_scores(
        self, vehicle_id: int, top_n: int, model_name: str = "user_similarity"
    ) -> List[Dict]:
        """Return only similarity scores without enrichment."""
        pass

class ICollaborativeRecommender(ABC):
    @abstractmethod
    async def get_collaborative_recommendations(
        self, user_id: int, top_n: int
    ) -> Dict[int, float]:
        """Generate recommendations using collaborative filtering"""
        pass

class IHybridRecommender(ABC):
    @abstractmethod
    async def get_recommendations(
        self, user_id: int, top_n: int
    ) -> RecommendationResponse:
        """Combine content-based and collaborative filtering into hybrid recommendations."""
        pass

class IScoreCombiner(ABC):
    @abstractmethod
    def combine(
        self,
        content_scores: Dict[int, float],
        collaborative_scores: Dict[int, float],
        content_weight: float = 0.5,
        collaborative_weight: float = 0.5,
    ) -> Dict[int, float]:
        """Combine content-based and collaborative scores into a hybrid score dict"""
        pass