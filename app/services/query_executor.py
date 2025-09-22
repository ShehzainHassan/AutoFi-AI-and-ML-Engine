import logging
import re
from typing import Any, List, Dict
import sqlparse

logger = logging.getLogger(__name__)
MAX_ROWS = 10

RELEVANT_TABLES = {
    "Vehicles": ["Id", "Make", "Model", "Year", "Price", "Mileage", "Color", "Transmission", "FuelType"],
    "Auctions": ["AuctionId", "VehicleId", "StartUtc", "EndUtc", "StartingPrice", "CurrentPrice", "Status", "CreatedUtc", "UpdatedUtc", "ScheduledStartTime", "PreviewStartTime", "IsReserveMet"],
    "Bids": ["BidId", "AuctionId", "UserId", "Amount", "IsAuto", "CreatedUtc"],
    "AutoBids": ["Id","UserId", "AuctionId", "MaxBidAmount", "CurrentBidAmount", "IsActive", "BidStrategyType", "CreatedAt", "UpdatedAt", "ExecutedAt"],
    "BidStrategies": ["AuctionId", "UserId", "Type", "BidDelaySeconds", "MaxBidsPerMinute", "MaxSpreadBids", "PreferredBidTiming", "CreatedAt", "UpdatedAt"],
    "Users": ["Id", "Name", "Email", "CreatedUtc", "LastLoggedIn"],
    "UserSavedSearches": ["UserId", "Search"],
    "UserInteractions": ["Id", "UserId", "VehicleId", "InteractionType", "CreatedAt"],
    "Watchlists": ["WatchlistId", "UserId", "AuctionId", "CreatedUtc"],
    "VehicleFeatures": ["Make", "Model", "Drivetrain", "Engine", "FuelEconomy", "Performance", "Measurements", "Options"],
}

class QueryExecutor:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def _is_safe_select(self, query: str) -> bool:
        parsed = sqlparse.parse(query)
        if not parsed:
            return False
        stmt = parsed[0]
        return stmt.get_type() == "SELECT"

    def _ensure_limit(self, query: str) -> str:
        lowered = query.lower()
        if "limit" not in lowered:
            query = query.rstrip().rstrip(";")
            query += f" LIMIT {MAX_ROWS}"
        return query

    @staticmethod
    def enforce_schema(query: str) -> str:
        for table, cols in RELEVANT_TABLES.items():
            query = re.sub(
                rf'(?<!")\b{table}\b(?!")',
                f'"{table}"',
                query,
                flags=re.IGNORECASE
            )
            for col in cols:
                query = re.sub(
                    rf'(?<!")\b{col}\b(?!")',
                    f'"{col}"',
                    query,
                    flags=re.IGNORECASE
                )
        return query

    def _check_user_filters(self, query: str, user_context: Dict[str, Any]) -> None:
        """
        Ensure that filters on Users.Id, Users.Name, Users.Email
        match the current user_context. If not, raise exception.
        """
        if not user_context:
            return  # nothing to check

        user_id = str(user_context.get("user_id", "")).lower()
        user_name = str(user_context.get("name", "")).lower()
        user_email = str(user_context.get("email", "")).lower()

        # Check for Users."Id"
        match_id = re.search(r'"Users"\."Id"\s*=\s*(\d+)', query, flags=re.IGNORECASE)
        if match_id:
            val = match_id.group(1).lower()
            if val != user_id:
                raise ValueError("Unauthorized access: UserId filter does not match context")

        # Check for Users."Name"
        match_name = re.search(r'"Users"\."Name"\s*=\s*\'([^\']+)\'', query, flags=re.IGNORECASE)
        if match_name:
            val = match_name.group(1).lower()
            if val != user_name:
                raise ValueError("Unauthorized access: User Name filter does not match context")

        # Check for Users."Email"
        match_email = re.search(r'"Users"\."Email"\s*=\s*\'([^\']+)\'', query, flags=re.IGNORECASE)
        if match_email:
            val = match_email.group(1).lower()
            if val != user_email:
                raise ValueError("Unauthorized access: User Email filter does not match context")

    async def execute_safe_query(
        self, query: str, user_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        try:
            if not self._is_safe_select(query):
                raise ValueError("Only SELECT queries are allowed")

            safe_query = self.enforce_schema(query)
            safe_query = self._ensure_limit(safe_query)

            # Enforce user-specific access rules
            self._check_user_filters(safe_query, user_context)

            logger.debug(f"Executing query: {safe_query}")

            async with self.db_manager.get_connection() as conn:
                rows = await conn.fetch(safe_query)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return [{"error": "Database query execution failed."}]
