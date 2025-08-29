from fastapi import Request
from app.services.user_context_service import MLUserContextService

def get_ml_service(request: Request) -> MLUserContextService:
    """
    Dependency provider for MLUserContextService using the app's DependencyContainer.
    """
    return request.app.state.container.get(MLUserContextService)
