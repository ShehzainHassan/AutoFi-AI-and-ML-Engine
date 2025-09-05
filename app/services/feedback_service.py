from app.db import DatabaseManager
from app.schemas.ai_schemas import FeedbackEnum
from app.exceptions.feedback_exceptions import MessageNotFoundError
import logging

logger = logging.getLogger(__name__)

class FeedbackService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def submit_feedback(self, message_id: int, vote: FeedbackEnum):
        logger.info(f"submit_feedback called with message_id={message_id}, vote={vote}")
        async with self.db_manager.get_connection() as conn:
            current_feedback = await conn.fetchval(
                'SELECT "Feedback" FROM "ChatMessages" WHERE "Id" = $1',
                message_id
            )
            logger.info(f"current_feedback: {current_feedback}")
            if current_feedback is None:
                raise MessageNotFoundError(message_id)

            new_feedback = vote.value
            if current_feedback == vote.value:
                new_feedback = FeedbackEnum.NOT_VOTED.value

            await conn.execute(
                'UPDATE "ChatMessages" SET "Feedback" = $1 WHERE "Id" = $2',
                new_feedback, message_id
            )
            logger.info(f"Feedback updated to: {new_feedback}")
            return new_feedback
