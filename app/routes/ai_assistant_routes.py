from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
from fastapi.encoders import jsonable_encoder
import logging
import asyncio
from app.schemas.ai_schemas import EnrichedAIQuery
from app.schemas.ai_schemas import AIQueryRequest, AIResponseModel
from app.services.ai_assistant_service import AIQueryService
from app.services.query_executor import QueryExecutor
from app.exceptions.recommendation_exceptions import UserNotFoundError
from app.security.auth_middleware import AuthService
from config.app_config import settings
from app.middleware.rate_limit_middleware import limiter
from app.services.user_context_service import MLUserContextService
from app.dependencies.ai_dependencies import get_ml_service
import simplejson as json
from datetime import datetime

router = APIRouter(tags=["AI Assistant"])
logger = logging.getLogger(__name__)
security = HTTPBearer()
auth_service = AuthService(jwt_secret=settings.JWT_SECRET, jwt_algorithm=settings.JWT_ALGORITHM)

async def get_ai_service(request: Request):
    db_manager = request.app.state.container.db_manager
    async with db_manager.get_connection() as conn:
        executor = QueryExecutor(conn)
        service = AIQueryService(
            api_key=settings.OPENAI_API_KEY,
            query_executor=executor
        )
        yield service

@router.post("/query", response_model=AIResponseModel)
@limiter.limit("10/minute")
async def ai_query(
    request: Request,
    payload: EnrichedAIQuery,
    service: AIQueryService = Depends(get_ai_service),
    current_user: dict = Depends(auth_service.verify_token),
    ml_service: MLUserContextService = Depends(get_ml_service),
):
    body = await request.body()
    correlation_id = request.headers.get("x-correlation-id", "N/A")
    logger.info(f"Raw request body: {body.decode()}")

    ml_context = await ml_service.get_ml_context(payload.query.user_id)

    logger.info(
        f"Received AI query. CorrelationId={correlation_id}, UserId={payload.query.user_id}, Question={payload.query.question}"
    )

    def json_safe(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    logger.info(f".NET Context: {json.dumps(payload.context, indent=2, default=json_safe)}")
    logger.info(f"ML Context: {json.dumps(ml_context, indent=2, default=json_safe)}")

    # Authorization check
    if current_user.get("user_id") != payload.query.user_id and not current_user.get("is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not authorized")

    db_manager = request.app.state.container.db_manager

    try:
        async with db_manager.get_connection() as conn:
            # Validate user existence
            user_exists = await conn.fetchval(
                'SELECT 1 FROM "Users" WHERE "Id"=$1', payload.query.user_id
            )
            if not user_exists:
                raise UserNotFoundError(payload.query.user_id)

            # Merge labeled contexts
            combined_context = {
                "dotnet_context": payload.context or {},
                "ml_context": ml_context or {}
            }

            # Generate AI response
            loop = asyncio.get_running_loop()
            response: AIResponseModel = await loop.run_in_executor(
                None, lambda: service.generate_response_sync(
                    user_query=payload.query.question,
                    user_id=payload.query.user_id,
                    context=combined_context
                )
            )

            logger.debug(
                f"Response fields: ui_type={response.ui_type}, query_type={response.query_type}, "
                f"actions={response.suggested_actions}, sources={response.sources}"
            )
            return jsonable_encoder(response)

    except UserNotFoundError as e:
        logger.warning(f"User not found: {e.message}")
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"AI query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process AI query.")

@router.get("/context/{user_id}")
async def get_ml_user_context(user_id: int, ml_service: MLUserContextService = Depends(get_ml_service)):
    context = await ml_service.get_ml_context(user_id)
    if not context:
        raise HTTPException(status_code=404, detail="User ML context not found")
    return context

# @router.post("/feedback")
# async def submit_feedback(feedback: AIQueryFeedback, request: Request):
# TO DO

# @router.get("/popular-queries", response_model=List[str])
# async def popular_queries(request: Request):
# TO DO