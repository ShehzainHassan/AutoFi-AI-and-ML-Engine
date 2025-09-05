import logging
from typing import Optional
from fastapi.encoders import jsonable_encoder
import simplejson as json

from app.services.schema_provider import get_schema_context
from app.services.query_executor import QueryExecutor
from app.schemas.ai_schemas import AIResponseModel, UIType
from app.utils.context_formatter import format_context_for_prompt
from app.utils.openai_client import OpenAIClient
from app.utils.response_parser import extract_json_block, clean_answer_text
from app.utils.ui_block_builder import UIBlockBuilder
from app.utils.answer_generator import AnswerGenerator
logger = logging.getLogger(__name__)


class AIQueryService:
    CLASSIFICATION_PROMPT = """
You are BoxAssistant, an AI assistant for AutoFi.
Classify the user's query strictly into ONE of these categories:
GENERAL, VEHICLE_SEARCH, AUCTION_QUERY, FINANCE_CALC, USER_SPECIFIC

Return only the classification as plain text.
Do NOT add any explanation or formatting.

User query: {user_query}
"""

    GENERAL_PROMPT = """
You are BoxAssistant, an AI assistant for AutoFi.
Answer the user’s query using general knowledge.
Respond ONLY with valid JSON. Do NOT wrap it in Markdown or add explanations.

Return JSON with fields:
- answer: human-friendly text
- ui_type: one of TEXT, TABLE, CARD_GRID, CALCULATOR, CHART
- query_type: GENERAL
- data: structured info if applicable, else []
- suggested_actions: 2-3 follow-up questions
- sources: list of website URLs or [] if none
- sql: null

User query: {user_query}
"""

    STRUCTURED_PROMPT = """
You are BoxAssistant, an AI assistant for AutoFi.
Generate a structured response using the database schema.

Return JSON with fields:
- answer: human-friendly summary (do NOT include SQL here)
- ui_type: one of TEXT, TABLE, CARD_GRID, CALCULATOR, CHART
- query_type: VEHICLE_SEARCH, AUCTION_QUERY, FINANCE_CALC, or USER_SPECIFIC
- data: {{}}
- suggested_actions: 2-3 follow-up questions
- sources: null or []
- sql: valid SQL query (for VEHICLE_SEARCH, AUCTION_QUERY, FINANCE_CALC). null if USER_SPECIFIC.

Database schema:
{schema_context}

User query: {user_query}
User context: {user_context}
"""

    def __init__(self, openai_client: OpenAIClient, query_executor: QueryExecutor):
        self.query_executor = query_executor
        self.openai_client = openai_client
        self.schema_context = get_schema_context()

    async def generate_response(
        self,
        user_query: str,
        user_id: int,
        context: Optional[dict] = None,
    ) -> AIResponseModel:
        """
        Generate a structured AI response with safe classification, parsing,
        and actual DB execution if SQL is returned.
        """

        # --- Classification ---
        classification_prompt = self.CLASSIFICATION_PROMPT.format(user_query=user_query)
        query_type_raw = self.openai_client.call_openai_with_retry(classification_prompt)

        query_type = "GENERAL"
        try:
            parsed_cls = None
            try:
                parsed_cls = json.loads(query_type_raw)
            except Exception:
                pass

            if isinstance(parsed_cls, dict):
                query_type = parsed_cls.get("CLASSIFICATION", "").strip().upper()
            else:
                query_type = str(query_type_raw).strip().upper()
        except Exception as e:
            logger.warning(f"[BoxAssistant] Failed to parse classification: {e}")
            query_type = "GENERAL"

        valid_types = {"GENERAL", "VEHICLE_SEARCH", "AUCTION_QUERY", "FINANCE_CALC", "USER_SPECIFIC"}
        if query_type not in valid_types:
            query_type = "GENERAL"

        # --- Prompt Selection ---
        if query_type == "GENERAL":
            prompt = self.GENERAL_PROMPT.format(user_query=user_query)
        else:
            user_context_str = format_context_for_prompt(context) if context else ""
            prompt = self.STRUCTURED_PROMPT.format(
                schema_context=self.schema_context,
                user_query=user_query,
                user_context=user_context_str,
            )

        # --- Get AI Response ---
        raw_response = self.openai_client.call_openai_with_retry(prompt)
        logger.info(f"[BoxAssistant] Raw response: {raw_response}")

        # --- Parse JSON ---
        parsed = {}
        try:
            clean_raw = extract_json_block(raw_response)
            parsed = json.loads(clean_raw)
        except Exception as e:
            logger.error(f"[BoxAssistant] Failed to parse AI JSON: {e}")
            parsed = {}

        # --- Extract fields ---
        answer = clean_answer_text(parsed.get("answer", "Sorry, I could not understand your question."))
        ui_type_raw = parsed.get("ui_type", "TEXT")

        try:
            ui_type = UIType(ui_type_raw.upper())
        except Exception:
            ui_type = UIType.TEXT

        sources = parsed.get("sources", []) if query_type == "GENERAL" else []
        sql_query = parsed.get("sql") if query_type in {"VEHICLE_SEARCH", "AUCTION_QUERY", "FINANCE_CALC"} else None
        data = parsed.get("data", {} if query_type != "GENERAL" else [])
        suggested_actions = parsed.get("suggested_actions", []) or []

        # --- Execute SQL if available ---
        if sql_query:
            logger.info(f"[BoxAssistant] Executing SQL: {sql_query}")
            
            db_result = await self.query_executor.execute_safe_query(sql_query)

            data = db_result
            if data:
                row_count = len(data) if isinstance(data, list) else 1
                answer = f"{answer} (showing {row_count} result(s))"
            else:
                answer = "Sorry, I couldn't find any matching results."
        
        final_answer = AnswerGenerator.generate(data, answer)
        ui_block = UIBlockBuilder.build(ui_type.value, data, final_answer)
        
        # --- Return response ---
        return AIResponseModel(
            answer=final_answer,
            ui_type=ui_type,
            query_type=query_type,
            data=jsonable_encoder(data),
            suggested_actions=suggested_actions,
            sources=sources,
            ui_block=ui_block
        )
