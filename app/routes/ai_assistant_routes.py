from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from typing import List
from fastapi.security import HTTPBearer
import logging
from app.schemas.ai_schemas import AIResponseModel, EnrichedAIQuery, FeedbackVote, PopularQueryDTO
from app.exceptions.recommendation_exceptions import UserNotFoundError
from app.security.auth_middleware import AuthService
from config.app_config import settings
from app.middleware.rate_limit_middleware import limiter
from app.dependencies.ai_dependencies import get_assistant_orchestrator
from app.exceptions.feedback_exceptions import MessageNotFoundError
from fastapi import BackgroundTasks
from app.interfaces.assistant_interfaces import IAssistantOrchestrator

router = APIRouter(tags=["AI Assistant"])
logger = logging.getLogger(__name__)
security = HTTPBearer()
auth_service = AuthService(jwt_secret=settings.JWT_SECRET, jwt_algorithm=settings.JWT_ALGORITHM)

@router.post("/query", response_model=AIResponseModel)
@limiter.limit("10/minute")
async def ai_query(
    request: Request,
    payload: EnrichedAIQuery,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(auth_service.verify_token),
    orchestrator: IAssistantOrchestrator = Depends(get_assistant_orchestrator),
):
    if current_user.get("user_id") != payload.query.user_id and not current_user.get("is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not authorized")

    try:
        response = await orchestrator.handle_query(user_id=payload.query.user_id, question=payload.query.question, context=payload.context)
        background_tasks.add_task(orchestrator.save_popular_query, payload.query.question, request.app.state.container.db_manager)
        return response
    except UserNotFoundError as e:
        logger.warning(f"User not found: {e.message}")
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"AI query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process AI query.")

@router.get("/context/{user_id}")
async def get_ml_user_context(user_id: int, orchestrator: IAssistantOrchestrator = Depends(get_assistant_orchestrator)):
    context = await orchestrator.get_user_context(user_id)
    if not context:
        raise HTTPException(status_code=404, detail="User ML context not found")
    print(context)
    return context

@router.post("/feedback")
@limiter.limit("10/minute")
async def submit_feedback(
    request: Request,
    feedback: FeedbackVote,
    orchestrator: IAssistantOrchestrator = Depends(get_assistant_orchestrator),
):
    try:
        updated_vote = await orchestrator.submit_feedback(
            message_id=feedback.message_id,
            vote=feedback.vote
        )
        return {"message_id": feedback.message_id, "feedback": updated_vote}

    except MessageNotFoundError as e:
        logger.warning(f"MessageNotFoundError: {e.message} | Code: {e.error_code}")
        raise HTTPException(status_code=404, detail={"error": e.message, "code": e.error_code})

    except Exception as e:
        logger.exception(f"Unhandled exception while submitting feedback for message_id={feedback.message_id}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@router.get("/popular-queries", response_model=List[PopularQueryDTO])
async def popular_queries(limit: int = Query(10, ge=1, le=50), orchestrator: IAssistantOrchestrator = Depends(get_assistant_orchestrator)):
    top = await orchestrator.get_popular_queries(limit)
    return top