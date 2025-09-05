import logging
import re
from typing import Any, Dict, List
from app.services.schema_provider import RELEVANT_TABLES

logger = logging.getLogger(__name__)
MAX_ROWS = 10

class QueryExecutor:
    def __init__(self, db_manager):
        self.db_manager = db_manager  

    @staticmethod
    def enforce_schema(query: str) -> str:
        """
        Ensure all table and column names match schema casing & quoting,
        without double-quoting things already quoted.
        """
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



    async def execute_safe_query(
        self, query: str, params: Dict[str, Any] = {}
    ) -> List[Dict[str, Any]]:
        if not self.validate_query(query):
            logger.warning(f"Query blocked due to unsafe content: {query}")
            return [{"error": "Unsafe query detected."}]

        query = self.enforce_schema(query)

        lowered = query.lower()
        if "limit" not in lowered and "count(" not in lowered:
            query = query.rstrip().rstrip(";") + f" LIMIT {MAX_ROWS}"

        try:
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetch(query, *params.values())
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return [{"error": "Database query execution failed."}]


    def validate_query(self, query: str) -> bool:
        stripped_query = query.strip()
        lowered_query = stripped_query.lower()

        if not lowered_query.startswith("select"):
            return False

        if ";" in stripped_query[:-1]:
            logger.warning("Forbidden: multiple SQL statements detected")
            return False

        # Block dangerous keywords
        forbidden_keywords = [
            "drop", "delete", "alter", "insert", "update",
            "--", "exec", "truncate"
        ]
        for kw in forbidden_keywords:
            if kw in lowered_query:
                logger.warning(f"Forbidden keyword found: {kw}")
                return False

        tables_in_query = (
            re.findall(r'\bfrom\s+"?(\w+)"?', stripped_query, re.IGNORECASE)
            + re.findall(r'\bjoin\s+"?(\w+)"?', stripped_query, re.IGNORECASE)
        )
        tables_in_query = [t.strip('"') for t in tables_in_query]

        # Validate tables against RELEVANT_TABLES
        for table in tables_in_query:
            if table not in RELEVANT_TABLES:
                logger.warning(f"Table not allowed: {table}")
                return False

        return True


