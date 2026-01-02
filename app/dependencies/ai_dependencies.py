from fastapi import Request, HTTPException, status
from app.services.user_context_service import MLUserContextService
from app.services.feedback_service import FeedbackService
from app.services.ai_assistant_service import AIQueryService
from app.interfaces.assistant_interfaces import IAssistantOrchestrator
from config.app_config import settings

def get_ml_service(request: Request) -> MLUserContextService:
    """
    Dependency provider for MLUserContextService using the app's DependencyContainer.
    """
    container = getattr(request.app.state, 'container', None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing. Please try again in a moment."
        )
    return container.get(MLUserContextService)

def get_feedback_service(request: Request) -> FeedbackService:
    """
    Dependency provider for FeedbackService using the app's DependencyContainer.
    """
    container = getattr(request.app.state, 'container', None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing. Please try again in a moment."
        )
    return container.get(FeedbackService)

def get_ai_service(request: Request) -> AIQueryService:
    """
    Dependency provider for AIQueryService using the app's DependencyContainer.
    """
    container = getattr(request.app.state, 'container', None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing. Please try again in a moment."
        )
    return container.get(AIQueryService)


def get_assistant_orchestrator(request: Request) -> IAssistantOrchestrator:
    container = getattr(request.app.state, 'container', None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing. Please try again in a moment."
        )
    return container.get(IAssistantOrchestrator)

def check_ai_enabled():
    if not settings.AI_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI functionality is currently disabled."
        )