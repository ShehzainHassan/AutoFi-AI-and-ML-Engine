from typing import Dict, Any
from app.db import DatabaseManager
from app.services.caching_service import CachingService

class MLUserContextService:
    """
    Fetch ML-specific context for FastAPI.
    Uses Redis caching to reduce database hits.
    """

    def __init__(self, db: DatabaseManager, cache: CachingService):
        self.db = db
        self.cache = cache

    async def get_ml_context(self, user_id: int) -> Dict[str, Any]:
        cached = await self.cache.get_cached_ml_context(user_id)
        if cached:
            return cached

        async with self.db.get_connection() as conn:
            interactions_query = """
                SELECT "VehicleId", "InteractionType", "CreatedAt"
                FROM "UserInteractions"
                WHERE "UserId" = $1
                ORDER BY "CreatedAt" DESC
                LIMIT 5
            """
            events_query = """
                SELECT "EventType", "AuctionId", "CreatedAt"
                FROM "AnalyticsEvents"
                WHERE "UserId" = $1
                ORDER BY "CreatedAt" DESC
                LIMIT 5
            """
            interactions = await conn.fetch(interactions_query, user_id)
            events = await conn.fetch(events_query, user_id)

        context = {
            "user_interactions": [dict(i) for i in interactions],
            "analytics_events": [dict(e) for e in events]
        }

        await self.cache.set_cached_ml_context(user_id, context)

        return context
