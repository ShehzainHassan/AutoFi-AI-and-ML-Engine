import pandas as pd
from typing import List, Dict

class UserRepository:
    def __init__(self, pool):
        self.pool = pool
        self._interactions_df: pd.DataFrame | None = None 

    async def user_exists(self, user_id: int) -> bool:
        query = 'SELECT EXISTS (SELECT 1 FROM "Users" WHERE "Id" = $1)'
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, user_id)

    async def load_interactions_summary(self) -> pd.DataFrame:
        """Load all user interactions into a DataFrame (async DB)."""
        if self._interactions_df is not None:
            return self._interactions_df

        query = """
            SELECT "UserId" AS user_id, "VehicleId" AS vehicle_id,
                   "InteractionType" AS interaction_type, COUNT(*) AS count
            FROM "UserInteractions"
            GROUP BY "UserId", "VehicleId", "InteractionType"
            ORDER BY user_id, vehicle_id;
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            rows_list = [dict(row) for row in rows]

        df = pd.DataFrame(rows_list)
        self._interactions_df = df
        return df

    async def get_user_interactions(self, user_id: int) -> List[Dict[str, float]]:
        """
        Fetch user interactions from the DB.
        """
        query = """
            SELECT "VehicleId" AS vehicle_id, COUNT(*) AS weight
            FROM "UserInteractions"
            WHERE "UserId" = $1
            GROUP BY "VehicleId"
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, user_id)
        return [dict(row) for row in rows]