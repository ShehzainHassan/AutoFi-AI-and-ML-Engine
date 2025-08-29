from typing import Any, Dict, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging
import re

logger = logging.getLogger(__name__)

class QueryExecutor:
    def __init__(self, conn):
        self.conn = conn

    async def execute_safe_query(self, query: str, params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """
        Execute SQL query safely using parameterization.
        Supports AI-generated SELECT queries across multiple tables.
        """
        if not self.validate_query(query):
            logger.warning(f"Query blocked due to unsafe keywords: {query}")
            return [{"error": "Unsafe query detected."}]

        if not query.strip().lower().startswith("select"):
            logger.warning("Only SELECT queries are allowed.")
            return [{"error": "Only SELECT queries are allowed."}]

        try:
            result = await self.conn.fetch(query, *params.values())
            rows = [dict(row) for row in result]
            return rows
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return [{"error": "Database query execution failed."}]

    def validate_query(self, query: str) -> bool:
        """
        Validate query to prevent unsafe operations.
        Only allow SELECT queries and safe tables.
        """
        forbidden_keywords = ["DROP", "DELETE", "ALTER", "INSERT", "UPDATE", "--", ";", "EXEC", "TRUNCATE"]
        upper_query = query.upper()
        for kw in forbidden_keywords:
            if kw in upper_query:
                logger.warning(f"Forbidden keyword found in query: {kw}")
                return False

        allowed_tables = [
            "Vehicles", "Auctions", "Bids", "AutoBids", "BidStrategies",
            "Users", "UserSavedSearches", "UserInteractions", "Watchlists", "AuctionAnalytics", "AnalyticsEvents"
        ]
        tables_in_query = re.findall(r"\bFROM\s+(\w+)", query, re.IGNORECASE)
        tables_in_query += re.findall(r"\bJOIN\s+(\w+)", query, re.IGNORECASE)

        for table in tables_in_query:
            if table not in allowed_tables:
                logger.warning(f"Table not allowed in query: {table}")
                return False

        return True
