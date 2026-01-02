from fastapi import APIRouter, Request
from app.db import DatabaseManager
from config.app_config import settings
from fastapi import Depends
from app.dependencies.dependency_container import DependencyContainer

router = APIRouter()

@router.get("/health")
async def health(request: Request):
    """
    Health check for database, ML service, and orchestrator readiness
    """
    container = getattr(request.app.state, 'container', None)
    db_manager: DatabaseManager = container.db_manager if container else None

    orchestrator_ready = False
    ml_ready = False

    if container:
        try:
            orchestrator_ready = container.orchestrator is not None
        except Exception:
            pass

        try:
            ml_service = getattr(container, "ml_service", None)
            ml_ready = getattr(ml_service, "models_loaded", False) if ml_service else False
        except Exception:
            pass

    return {
        "db": db_manager.pool is not None if db_manager else False,
        "ml_models_loaded": ml_ready,
        "orchestrator_ready": orchestrator_ready,
        "version": request.app.version,
        "status": "ready" if container else "initializing"
    }
