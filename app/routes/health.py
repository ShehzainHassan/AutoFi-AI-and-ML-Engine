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
    db_manager: DatabaseManager = request.app.state.container.db_manager if request.app.state.container else None

    orchestrator_ready = False
    ml_ready = False

    if request.app.state.container:
        try:
            orchestrator_ready = request.app.state.container.orchestrator is not None
        except Exception:
            pass

        try:
            ml_service = getattr(request.app.state.container, "ml_service", None)
            ml_ready = getattr(ml_service, "models_loaded", False)
        except Exception:
            pass

    return {
        "db": db_manager.pool is not None if db_manager else False,
        "ml_models_loaded": ml_ready,
        "orchestrator_ready": orchestrator_ready,
        "version": request.app.version,
    }
