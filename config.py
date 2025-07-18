import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database configuration - Railway provides DATABASE_URL
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/autofi")
    
    # Model configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "trained_models/")
    MAX_RECOMMENDATIONS: int = int(os.getenv("MAX_RECOMMENDATIONS", "10"))
    
    # Server configuration
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()