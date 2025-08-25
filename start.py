"""
Railway deployment startup script for AutoFi Vehicle Recommendation System
"""
import os
import uvicorn
from config.app_config import settings

def main():
    """Start the FastAPI application with Railway-optimized settings"""
    # Railway provides PORT environment variable
    port = int(os.getenv("PORT", settings.PORT))
    host = settings.HOST
    
    print(f"Starting AutoFi Vehicle Recommendation System on {host}:{port}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Database URL configured: {'Yes' if settings.DATABASE_URL else 'No'}")
    
    # Start the application
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        # Production settings for Railway
        workers=1,  # Railway handles scaling
        loop="asyncio",
        http="auto"
    )

if __name__ == "__main__":
    main() 