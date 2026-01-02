from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status, Path, Request
from app.middleware.rate_limit_middleware import limiter
from app.schemas.schemas import RecommendationResponse, SimilarVehiclesResponse
from app.interfaces.recommendation_interfaces import IRecommendationOrchestrator
from app.exceptions.recommendation_exceptions import (
    UserNotFoundError,
    RecommendationServiceError,
    InsufficientDataError,
    ModelNotAvailableError,
    VehicleNotFoundError,
)
from app.security.auth_middleware import AuthService
from config.app_config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
auth_service = AuthService(jwt_secret=settings.JWT_SECRET, jwt_algorithm=settings.JWT_ALGORITHM)

def get_orchestrator(request: Request) -> IRecommendationOrchestrator:
    """Get orchestrator from app state, with proper error handling."""
    container = getattr(request.app.state, 'container', None)
    if container is None:
        logger.error("Container not initialized - application may still be starting up")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing. Please try again in a moment."
        )
    if not hasattr(container, 'orchestrator') or container.orchestrator is None:
        logger.error("Orchestrator not available in container")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is not ready"
        )
    return container.orchestrator


@router.get("/user/{user_id}", response_model=RecommendationResponse)
@limiter.limit("10/minute")
async def get_recommendations(
    request: Request,
    user_id: int = Path(..., ge=1, le=2147483647),
    top_n: int = Query(default=5, ge=1, le=50),
    current_user: dict = Depends(auth_service.verify_token),
    orchestrator: IRecommendationOrchestrator = Depends(get_orchestrator),
):
    if current_user.get("user_id") != user_id and not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other user's recommendations",
        )

    try:
        return await orchestrator.get_recommendations(user_id, top_n)
    except UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except InsufficientDataError as e:
        raise HTTPException(status_code=422, detail=e.message)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=e.message)
    except RecommendationServiceError as e:
        logger.error(f"Recommendation service error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/similar/{vehicle_id}", response_model=SimilarVehiclesResponse)
@limiter.limit("10/minute")
async def get_similar_vehicles(
    request: Request,
    vehicle_id: int,
    top_n: int = Query(default=5, ge=1, le=50),
    orchestrator: IRecommendationOrchestrator = Depends(get_orchestrator),
):
    try:
        return await orchestrator.get_similar_vehicles(vehicle_id, top_n)
    except VehicleNotFoundError:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=e.message)
    except RecommendationServiceError as e:
        logger.error(f"Recommendation error for vehicle {vehicle_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logger.exception(f"Unhandled error in similar vehicles endpoint: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")
