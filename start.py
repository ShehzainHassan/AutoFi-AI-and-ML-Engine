"""
Railway deployment startup script
"""
import os
import sys
import uvicorn
from pydantic import ValidationError
from config.app_config import Settings

def print_config_summary(settings: Settings):
    """Prints important environment variables and settings at startup."""
    print("\n=== AutoFi Startup Configuration ===")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Host: {settings.HOST}")
    print(f"Port: {os.getenv('PORT', settings.PORT)}")
    print(f"Database URL configured: {'Yes' if settings.DATABASE_URL else 'No'}")

    print("\n--- Redis ---")
    print(f"Redis Host: {settings.REDIS_HOST}")
    print(f"Redis DB: {settings.REDIS_DB}")

    print("\n--- ML Configuration ---")
    print(f"Model Path: {settings.MODEL_PATH}")
    print(f"Max Recommendations: {settings.MAX_RECOMMENDATIONS}")

    print("\n--- OpenAI ---")
    print(f"Model: {settings.OPENAI_MODEL}")
    print(f"Max Tokens: {settings.OPENAI_MAX_TOKENS}")
    print(f"Timeout: {settings.OPENAI_TIMEOUT}s")

    print("\n--- Feature Weights ---")
    print("Vehicle Feature Weights:", getattr(settings, "VEHICLE_FEATURE_WEIGHTS", None))
    print("User Feature Weights:", getattr(settings, "USER_FEATURE_WEIGHTS", None))
    print("Interaction Weights:", getattr(settings, "INTERACTION_WEIGHTS", None))
    print("====================================\n")

def main():
    """Start the FastAPI application with Railway-optimized settings."""
    try:
        settings = Settings()
        print("✅ Environment variables validated successfully.\n")
        print_config_summary(settings)

        port = int(os.getenv("PORT", settings.PORT))
        host = settings.HOST

        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            workers=1,
            loop="asyncio",
            http="auto"
        )

    except ValidationError as e:
        print("❌ Environment validation failed:")
        print(e.json(indent=2))
        sys.exit(1)

    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()