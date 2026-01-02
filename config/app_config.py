from typing import Optional
from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    DB_POOL_MIN: PositiveInt = Field(default=5, ge=1)
    DB_POOL_MAX: PositiveInt = Field(default=20, ge=1)

    # Redis
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379, gt=0, le=65535)
    REDIS_DB: int = Field(default=0, ge=0)

    # Auth (optional - defaults provided for Railway deployment)
    JWT_SECRET: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_AUDIENCE: str = Field(default="AutoFiClient")

    # OpenAI (optional if AI_ENABLED is False)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = Field(default=5000, gt=0)
    OPENAI_TIMEOUT: float = Field(default=30.0, gt=0)
    OPENAI_TEMPERATURE: float = Field(default=0.2, gt=0)
    AI_ENABLED: bool = Field(default=True)

    # ML Model
    MODEL_PATH: str = Field(default="trained_models")
    MAX_RECOMMENDATIONS: PositiveInt = Field(default=10)

    # Server
    ENVIRONMENT: str = Field(default="production")
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000, gt=0, le=65535)
    

    # Config
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


settings = Settings()
