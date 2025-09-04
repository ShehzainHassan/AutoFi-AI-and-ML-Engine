import logging
from typing import Any, Dict, Optional
from app.services.schema_provider import get_schema_context
from app.services.query_executor import QueryExecutor
from app.schemas.ai_schemas import AIResponseModel
from fastapi.encoders import jsonable_encoder
import simplejson as json
from app.schemas.ai_schemas import UIType
from app.utils.context_formatter import format_context_for_prompt, json_safe
from app.utils.openai_client import OpenAIClient
from app.utils.response_parser import parse_ui_response
logger = logging.getLogger(__name__)
PROMPT_INSTRUCTION = "Return a SQL query if data is needed, else provide a personalized explanation."

class AIQueryService:
    def __init__(self, openai_client: OpenAIClient, query_executor: QueryExecutor):
        self.query_executor = query_executor
        self.openai_client = openai_client
        self.schema_context = get_schema_context()

    async def generate_response(
        self,
        user_query: str,
        user_id: int = None,
        context: Optional[dict] = None
    ) -> AIResponseModel:
        prompt = f"""
        You are BoxAssistant, an AI assistant for AutoFi.
        Use the following database schema and car features context to answer the user query:
        {self.schema_context}

        User query: {user_query}
        """
        if context:
            prompt += "\n\nUser Context Summary:\n"
            prompt += format_context_for_prompt(context)

        prompt += """
        Instructions:
        1. Determine the query type, which can be one of: "VEHICLE_SEARCH", "AUCTION_QUERY", "FINANCE_CALC", "USER_SPECIFIC", "GENERAL".
        2. Return a SQL query if data is needed, else provide a personalized explanation.
        3. Also generate 2 to 3 short, relevant follow-up questions the user might ask next in first-person style.
        4. If your answer is best displayed as a table, card grid, calculator, or chart, prepend the answer with one of these tags: [TABLE], [CARD_GRID], [CALCULATOR], [CHART]. Otherwise, use plain text.
        5. Identify sources you referenced in your answer.
        6. Return your response in JSON with the following keys:
        - "answer": string or SQL query (with UI tag if applicable)
        - "query_type": string (from the allowed types)
        - "suggested_actions": array of strings in first-person style
        - "sources": array of strings
        """

        raw_response = self.openai_client.call_openai_with_retry(prompt)
        logger.info(f"OpenAI returned: {raw_response}")

        try:
            parsed = json.loads(raw_response)
            main_answer = parsed.get("answer", "")
            suggested_actions = parsed.get("suggested_actions", [])
            query_type = parsed.get("query_type", "GENERAL")
            sources = parsed.get("sources", [])
        except Exception:
            main_answer = raw_response
            suggested_actions = ["Ask another question", "Show me similar options"]
            query_type = "GENERAL"
            sources = []

        is_sql = main_answer.strip().lower().startswith("select")
        if is_sql and self.query_executor.validate_query(main_answer):
            try:
                params: Dict[str, Any] = {}
                if user_id and query_type in ["USER_SPECIFIC", "AUCTION_QUERY", "FINANCE_CALC"]:
                    params["user_id"] = user_id
                # if params:
                #     rows = await self.query_executor.execute_safe_query(main_answer, params=params)
                # else:
                #     rows = await self.query_executor.execute_safe_query(main_answer)
                data = main_answer
                ui_type = {
                    "VEHICLE_SEARCH": UIType.CARD_GRID,
                    "AUCTION_QUERY": UIType.TABLE,
                    "USER_SPECIFIC": UIType.TABLE,
                    "FINANCE_CALC": UIType.CALCULATOR
                }.get(query_type, UIType.TEXT)
            except Exception as e:
                logger.error(f"SQL execution failed: {e}")
                data = {"error": "Failed to fetch data from database."}
                ui_type = UIType.TEXT
        else:
            ui_type, data = parse_ui_response(main_answer)

        answer_str = json.dumps(data, indent=2, default=json_safe, use_decimal=True)
        return AIResponseModel(
            answer=answer_str,
            ui_type=ui_type,
            query_type=query_type,
            data=jsonable_encoder(data),
            suggested_actions=suggested_actions,
            sources=sources,
        )
