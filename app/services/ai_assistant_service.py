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
from app.utils.query_classifier import classify_query

logger = logging.getLogger(__name__)


class AIQueryService:
    GENERAL_PROMPT = """
You are BoxAssistant, an AI assistant for AutoFi.
Answer the user's query using general knowledge.
Respond ONLY with valid JSON. Do NOT wrap it in Markdown or add explanations.
If the query contains any user name, nickname, or identifier, you must verify that it matches the current user's name or email exactly. If it refers to another user—even if the name is misspelled or ambiguous—respond with: "Sorry, I cannot assist with that."

Return JSON with fields:
- answer: human-friendly text
- ui_type: one of TEXT, TABLE, CARD_GRID, CALCULATOR, CHART
- chart_type: required if ui_type is CHART, must be "bar", "line", or "pie"
- data: structured info if applicable, else []
- suggested_actions: 2-3 follow-up questions
- sources: list of website URLs or [] if none
- sql: null

User query: {user_query}
"""

    STRUCTURED_PROMPT = """
You are BoxAssistant, an AI assistant for AutoFi.
Generate a structured response using the database schema.
Always mention Vehicle's Make, Model, Year when referring to vehicles.
When referring to Auction, always include Vehicle's Make, Model, Year of the vehicle being auctioned.
Ensure the SQL is syntactically valid and does not contain malformed fragments or newline artifacts.
Do not include literal characters like 'n' or broken aliases.
If ui_type is CHART, include a field chart_type with value "bar", "line", or "pie" only.
If the query contains any user name, nickname, or identifier, you must verify that it matches the current user's name or email exactly. If it refers to another user—even if the name is misspelled or ambiguous—respond then do not generate any sql"

Return JSON with fields:
- ui_type: one of TEXT, TABLE, CARD_GRID, CALCULATOR, CHART
- chart_type: required if ui_type is CHART, must be "bar", "line", or "pie"
- data: structured info if applicable, else []
- sources: null or []
- sql: valid SQL query

Do NOT include 'answer' or 'suggested_actions' in your response.

Database schema:
{schema_context}

User query: {user_query}
User context: {user_context}
"""

    DATA_SUMMARIZATION_PROMPT = """
You are BoxAssistant, an AI assistant for AutoFi.
Summarize the following structured data for the user query.
If the query contains any user name, nickname, or identifier, you must verify that it matches the current user's name or email exactly. If it refers to another user—even if the name is misspelled or ambiguous—respond then do not generate any answer"

Respond ONLY with valid JSON. Do NOT wrap it in Markdown or add explanations.

Return JSON with fields:
- answer: human-friendly summary of the data
- suggested_actions: 2-3 follow-up questions

User query: {user_query}
Structured data: {data}
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
        ml_context = context.get("ml_context") if context else {}
        query_type = await classify_query(user_query, ml_context)

        if query_type == "UNSAFE":
            return self._fallback_response("UNSAFE")

        if query_type == "GENERAL":
            prompt = self.GENERAL_PROMPT.format(user_query=user_query)
            raw_response = self.openai_client.call_openai_with_retry(prompt)
            logger.info(f"[BoxAssistant] Raw response: {raw_response}")

            try:
                clean_raw = extract_json_block(raw_response)
                parsed = json.loads(clean_raw)
                logger.info(f"[BoxAssistant] Parsed response: {json.dumps(parsed, indent=2)}")
            except Exception as e:
                logger.error(f"[BoxAssistant] Failed to parse AI JSON: {e}")
                return self._fallback_response(query_type)

            final_answer = clean_answer_text(parsed.get("answer", ""))
            ui_type = UIType(parsed.get("ui_type", "TEXT").upper())
            chart_type = parsed.get("chart_type") if ui_type == UIType.CHART else None
            data = parsed.get("data", [])
            sources = parsed.get("sources", [])
            suggested_actions = parsed.get("suggested_actions", [])
            ui_block = UIBlockBuilder.build(ui_type.value, data, final_answer, chart_type=chart_type)

            return AIResponseModel(
                answer=final_answer,
                ui_type=ui_type,
                query_type=query_type,
                data=jsonable_encoder(data),
                suggested_actions=suggested_actions,
                sources=sources,
                ui_block=ui_block,
                chart_type=chart_type
            )

        user_context_str = format_context_for_prompt(context) if context else ""
        prompt = self.STRUCTURED_PROMPT.format(
            schema_context=self.schema_context,
            user_query=user_query,
            user_context=user_context_str,
        )

        raw_response = self.openai_client.call_openai_with_retry(prompt)
        logger.info(f"[BoxAssistant] Raw response: {raw_response}")

        try:
            clean_raw = extract_json_block(raw_response)
            parsed = json.loads(clean_raw)
            logger.info(f"[BoxAssistant] Parsed response: {json.dumps(parsed, indent=2)}")
        except Exception as e:
            logger.error(f"[BoxAssistant] Failed to parse AI JSON: {e}")
            return self._fallback_response(query_type)

        ui_type = UIType(parsed.get("ui_type", "TEXT").upper())
        chart_type = parsed.get("chart_type") if ui_type == UIType.CHART else None
        sql_query = parsed.get("sql")
        data = parsed.get("data", {})
        sources = []
        suggested_actions = []
        final_answer = ""

        if not sql_query:
            if data:
                final_answer = ""
                print("CHART TYPE = ", chart_type)
                ui_block = UIBlockBuilder.build(ui_type.value, data, final_answer, chart_type=chart_type)
                return AIResponseModel(
                    answer=final_answer,
                    ui_type=ui_type,
                    query_type=query_type,
                    data=jsonable_encoder(data),
                    suggested_actions=[],
                    sources=[],
                    ui_block=ui_block,
                    chart_type=chart_type
                )
            else:
                logger.warning("[BoxAssistant] No SQL query generated and no data returned")
                return self._fallback_response(query_type)

        logger.info(f"[BoxAssistant] Executing SQL: {sql_query}")
        try:
            user_context = {
                "user_id": user_id,
                "name": context.get("name") if context else None,
                "email": context.get("email") if context else None,
            }

            db_result = await self.query_executor.execute_safe_query(sql_query, user_context=user_context)

            if isinstance(db_result, list) and db_result and "error" in db_result[0]:
                logger.warning(f"[BoxAssistant] Query rejected: {db_result[0]['error']}")
                return self._fallback_response(query_type)

            if not db_result:
                final_answer = "I could not find any matching results"
                data = []
            else:
                data = db_result
                safe_data = jsonable_encoder(data)
                summarization_prompt = self.DATA_SUMMARIZATION_PROMPT.format(
                    user_query=user_query,
                    data=json.dumps(safe_data, indent=2, default=str)
                )
                summary_response = self.openai_client.call_openai_with_retry(summarization_prompt)
                logger.info(f"[BoxAssistant] Summary response: {summary_response}")

                try:
                    summary_json = json.loads(extract_json_block(summary_response))
                    final_answer = clean_answer_text(summary_json.get("answer", ""))
                    suggested_actions = summary_json.get("suggested_actions", [])
                except Exception as e:
                    logger.error(f"[BoxAssistant] Failed to parse summary JSON: {e}")
                    final_answer = "Here are the results I found"

        except Exception as e:
            logger.error(f"[BoxAssistant] Query execution failed: {e}")
            return self._fallback_response(query_type)

        ui_block = UIBlockBuilder.build(ui_type.value, data, final_answer, chart_type=chart_type)

        return AIResponseModel(
            answer=final_answer,
            ui_type=ui_type,
            query_type=query_type,
            data=jsonable_encoder(data),
            suggested_actions=suggested_actions,
            sources=sources,
            ui_block=ui_block,
            chart_type=chart_type
        )

    def _fallback_response(self, query_type: str) -> AIResponseModel:
        fallback_text = "Sorry I cannot assist with that"
        return AIResponseModel(
            answer=fallback_text,
            ui_type=UIType.TEXT,
            query_type=query_type,
            data=[],
            suggested_actions=[],
            sources=[],
            ui_block=UIBlockBuilder.build("TEXT", [], fallback_text, chart_type=None),
            chart_type=None
        )