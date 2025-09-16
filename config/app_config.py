import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    DB_POOL_MIN: int
    DB_POOL_MAX: int

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int

    # Auth
    JWT_SECRET: str
    JWT_ALGORITHM: str
    JWT_AUDIENCE: str

    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    OPENAI_MAX_TOKENS: int
    OPENAI_TIMEOUT:str

    # ML Model
    MODEL_PATH: str
    MAX_RECOMMENDATIONS: int

    # Server
    HOST: str
    PORT: int

    # Environment
    ENVIRONMENT: str

    # Pydantic v2 style
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="allow")

settings = Settings()