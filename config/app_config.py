from typing import Annotated
from pydantic import Field, PositiveInt, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    DB_POOL_MIN: PositiveInt = Field(default=5, ge=1)
    DB_POOL_MAX: PositiveInt = Field(default=20, ge=1)

    # Redis
    REDIS_HOST: str
    REDIS_DB: int = Field(ge=0)

    # Auth
    JWT_SECRET: Annotated[str, StringConstraints(min_length=32)]
    JWT_ALGORITHM: str
    JWT_AUDIENCE: str

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = Field(default=5000, gt=0)
    OPENAI_TIMEOUT: float = Field(default=30.0, gt=0)
    AI_ENABLED: bool = True

    # ML Model
    MODEL_PATH: str
    MAX_RECOMMENDATIONS: PositiveInt = 10

    # Server
    ENVIRONMENT: str
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, gt=0, le=65535)

    # Config
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


settings = Settings()
