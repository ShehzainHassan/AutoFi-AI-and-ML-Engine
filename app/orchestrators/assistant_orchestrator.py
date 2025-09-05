import logging
from typing import Dict, List, Optional
from app.services.ai_assistant_service import AIQueryService
from app.services.user_context_service import MLUserContextService
from app.services.feedback_service import FeedbackService
from app.services.popular_query_service import PopularQueryService
from app.schemas.ai_schemas import AIResponseModel, PopularQueryDTO
from app.db import DatabaseManager
logger = logging.getLogger(__name__)

class AssistantOrchestrator:
    def __init__(
        self,
        ai_service: AIQueryService,
        ml_service: MLUserContextService,
        feedback_service: FeedbackService,
        popular_query_service: PopularQueryService,
        db_manager: DatabaseManager,
    ):
        self.ai_service = ai_service
        self.ml_service = ml_service
        self.feedback_service = feedback_service
        self.popular_query_service = popular_query_service
        self.db_manager = db_manager

    async def handle_query(self, user_id: int, question: str, context: Dict) -> AIResponseModel:
        ml_context = await self.ml_service.get_ml_context(user_id)

        combined_context = {
            "dotnet_context": context or {},
            "ml_context": ml_context or {},
        }

        response = await self.ai_service.generate_response(
            user_query=question,
            user_id=user_id,
            context=combined_context,
        )

        logger.info(f"AI Response generated for user={user_id}, question='{question}'")
        return response

    async def get_user_context(self, user_id: int) -> Dict:
        return await self.ml_service.get_ml_context(user_id)

    async def submit_feedback(self, message_id: str, vote: str) -> Dict:
        return await self.feedback_service.submit_feedback(message_id=message_id, vote=vote)

    async def get_popular_queries(self, conn, limit: int = 10) -> List[PopularQueryDTO]:
        async with self.db_manager.get_connection() as conn:
            return await self.popular_query_service.get_top_popular_queries(conn, limit)
        
    async def save_popular_query(self, question: str, db_manager: Optional[DatabaseManager] = None) -> Dict:
        try:
            return await self.popular_query_service.save_popular_query(
                question=question,
                db_manager=db_manager or self.db_manager
            )
        except Exception as e:
            logger.error(f"Failed to save popular query: {e}")
            return {"ok": False, "reason": "exception during save"}