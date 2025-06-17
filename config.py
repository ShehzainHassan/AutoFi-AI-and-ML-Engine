from pydantic import BaseSettings
class Settings(BaseSettings):
	DATABASE_URL: str = 'postgresql://user:pass@localhost:5432/autofi'
	MODEL_PATH: str = 'models/recommendation_model.pkl'
	MAX_RECOMMENDATIONS: int = 10

	class Config:
		env_file = ".env"

settings = Settings()