import json
from redis.asyncio import Redis
from typing import Optional
from app.schemas.schemas import RecommendationResponse
class CachingService:
    """
    Centralized Redis caching with consistent key schema and TTL handling.
    """

    def __init__(self, redis_client: Redis, default_ttl: int = 900):
        self.redis = redis_client
        self.default_ttl = default_ttl

    def _key_recommendations(self, user_id: int, top_n: int, model_type: str = "hybrid") -> str:
        return f"rec:user:{user_id}:top:{top_n}:model:{model_type}"

    def _key_vehicle_similarity(self, vehicle_id: int, top_n: int) -> str:
        return f"rec:vehicle:{vehicle_id}:top:{top_n}"

    async def get_cached_vehicle_similarity(self, vehicle_id: int, top_n: int) -> Optional[list[dict]]:
        key = self._key_vehicle_similarity(vehicle_id, top_n)
        cached = await self.redis.get(key)
        if not cached:
            return None
        import json
        return json.loads(cached)

    async def set_cached_vehicle_similarity(self, vehicle_id: int, top_n: int, recommendations: list[dict], ttl: Optional[int] = None) -> None:
        key = self._key_vehicle_similarity(vehicle_id, top_n)
        import json
        await self.redis.setex(key, ttl or self.default_ttl, json.dumps(recommendations))

    async def invalidate_vehicle_cache(self, vehicle_id: int) -> int:
        pattern = f"rec:vehicle:{vehicle_id}:*"
        deleted = 0
        async for key in self.redis.scan_iter(match=pattern):
            deleted += await self.redis.delete(key)
        return deleted
    
    async def get_cached_recommendations(self, user_id: int, top_n: int, model_type: str = "hybrid") -> Optional[RecommendationResponse]:
        key = self._key_recommendations(user_id, top_n, model_type)
        cached = await self.redis.get(key)
        if not cached:
            return None
        return RecommendationResponse.parse_raw(cached)

    async def set_cached_recommendations(self, user_id: int, top_n: int, recommendations: RecommendationResponse, model_type: str = "hybrid", ttl: Optional[int] = None) -> None:
        key = self._key_recommendations(user_id, top_n, model_type)
        await self.redis.setex(key, ttl or self.default_ttl, recommendations.json())

    async def invalidate_user_cache(self, user_id: int) -> int:
        """
        Invalidate all recommendation keys for a user. Returns number of deleted keys.
        """
        pattern = f"rec:user:{user_id}:*"
        deleted = 0
        async for key in self.redis.scan_iter(match=pattern):
            deleted += await self.redis.delete(key)
        return deleted

    def _key_ml_context(self, user_id: int) -> str:
        return f"context:user:{user_id}:ml"

    async def get_cached_ml_context(self, user_id: int) -> Optional[dict]:
        key = self._key_ml_context(user_id)
        cached = await self.redis.get(key)
        if not cached:
            return None
        return json.loads(cached)

    async def set_cached_ml_context(self, user_id: int, context: dict, ttl: Optional[int] = None) -> None:
        key = self._key_ml_context(user_id)
        await self.redis.setex(key, ttl or self.default_ttl, json.dumps(context, default=str))
