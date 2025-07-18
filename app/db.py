import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_db_connection():
    try:
        # First try to use DATABASE_URL (Railway's standard format)
        database_url = os.getenv("DATABASE_URL")
        
        if database_url:
            conn = psycopg2.connect(database_url)
            logger.info("Connected to database using DATABASE_URL")
            return conn
        
        # Fallback to individual connection parameters
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "autofi"),
            user=os.getenv("DB_USER", "user"),
            password=os.getenv("DB_PASSWORD", "pass"),
            port=os.getenv("DB_PORT", "5432")
        )
        logger.info("Connected to database using individual parameters")
        return conn
    
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
