from abc import ABC, abstractmethod
from typing import List, Dict
from app.schemas.ai_schemas import AIResponseModel, PopularQueryDTO
class IAssistantOrchestrator(ABC):
    """High-level orchestration for AI Assistant queries, context, and feedback."""

    @abstractmethod
    async def handle_query(self, user_id: int, question: str, context: Dict) -> "AIResponseModel":
        pass

    @abstractmethod
    async def get_user_context(self, user_id: int) -> Dict:
        pass

    @abstractmethod
    async def submit_feedback(self, message_id: str, vote: str) -> Dict:
        pass

    @abstractmethod
    async def get_popular_queries(self, limit: int = 10) -> List["PopularQueryDTO"]:
        pass
