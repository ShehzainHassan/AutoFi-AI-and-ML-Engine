import logging
import time
from typing import Optional
from fastapi.encoders import jsonable_encoder
import simplejson as json
from opentelemetry import trace

from app.services.schema_provider import get_schema_context
from app.services.query_executor import QueryExecutor
from app.schemas.ai_schemas import AIResponseModel, UIType
from app.utils.context_formatter import format_context_for_prompt
from app.utils.openai_client import OpenAIClient
from app.utils.response_parser import extract_json_block, clean_answer_text
from app.utils.ui_block_builder import UIBlockBuilder
from app.utils.query_classifier import classify_query
from app.utils.assistant_prompts import GENERAL_PROMPT, STRUCTURED_PROMPT, DATA_SUMMARIZATION_PROMPT
from app.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY, REQUEST_ERRORS

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("boxcars-ai")

class AIQueryService:
    def __init__(self, openai_client: OpenAIClient, query_executor: QueryExecutor):
        self.query_executor = query_executor
        self.openai_client = openai_client

    async def generate_response(
        self,
        user_query: str,
        user_id: int,
        context: Optional[dict] = None,
    ) -> AIResponseModel:
        ml_context = context.get("ml_context") if context else {}
        query_type = await classify_query(user_query, ml_context)

        REQUEST_COUNT.labels(endpoint="/api/ai/query", method="POST", status_code="200").inc()
        start_time = time.time()

        try:
            if query_type == "UNSAFE":
                REQUEST_ERRORS.labels(endpoint="/api/ai/query", method="POST").inc()
                return self._fallback_response("UNSAFE")

            with tracer.start_as_current_span("prompt_generation"):
                if query_type == "GENERAL":
                    prompt = GENERAL_PROMPT.format(user_query=user_query)
                else:
                    schema_context = get_schema_context()
                    user_context_str = format_context_for_prompt(context) if context else ""
                    prompt = STRUCTURED_PROMPT.format(
                        schema_context=schema_context,
                        user_query=user_query,
                        user_context=user_context_str,
                    )

            logger.debug(f"[BoxAssistant] Prompt length: {len(prompt)}")

            with tracer.start_as_current_span("openai_call"):
                raw_response, usage = self.openai_client.call_openai_with_retry(prompt, return_usage=True)

            logger.info(f"[BoxAssistant] Raw response: {raw_response}")
            if usage:
                logger.info(f"[BoxAssistant] Token usage: {usage.total_tokens} tokens")

            try:
                clean_raw = extract_json_block(raw_response)
                parsed = json.loads(clean_raw)
                logger.info(f"[BoxAssistant] Parsed response: {json.dumps(parsed, indent=2)}")
            except Exception as e:
                REQUEST_ERRORS.labels(endpoint="/api/ai/query", method="POST").inc()
                logger.error(f"[BoxAssistant] Failed to parse AI JSON: {e}")
                return self._fallback_response(query_type)

            ui_type = UIType(parsed.get("ui_type", "TEXT").upper())
            chart_type = parsed.get("chart_type") if ui_type == UIType.CHART else None
            sql_query = parsed.get("sql")
            data = parsed.get("data", {})
            sources = parsed.get("sources", []) or []
            suggested_actions = []
            final_answer = ""

            if query_type == "GENERAL":
                final_answer = clean_answer_text(parsed.get("answer", ""))
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

            if not sql_query:
                if data:
                    final_answer = ""
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
                    REQUEST_ERRORS.labels(endpoint="/api/ai/query", method="POST").inc()
                    logger.warning("[BoxAssistant] No SQL query generated and no data returned")
                    return self._fallback_response(query_type)

            logger.info(f"[BoxAssistant] Executing SQL: {sql_query}")
            with tracer.start_as_current_span("sql_execution"):
                user_context = {
                    "user_id": user_id,
                    "name": context.get("name") if context else None,
                    "email": context.get("email") if context else None,
                }
                db_result = await self.query_executor.execute_safe_query(sql_query, user_context=user_context)

            if isinstance(db_result, list) and db_result and "error" in db_result[0]:
                REQUEST_ERRORS.labels(endpoint="/api/ai/query", method="POST").inc()
                logger.warning(f"[BoxAssistant] Query rejected: {db_result[0]['error']}")
                return self._fallback_response(query_type)

            if not db_result:
                final_answer = "I could not find any matching results"
                data = []
            else:
                data = db_result
                safe_data = jsonable_encoder(data)
                summarization_prompt = DATA_SUMMARIZATION_PROMPT.format(
                    user_query=user_query,
                    data=json.dumps(safe_data, indent=2, default=str)
                )
                summary_response, _ = self.openai_client.call_openai_with_retry(summarization_prompt, return_usage=True)
                logger.info(f"[BoxAssistant] Summary response: {summary_response}")

                try:
                    summary_json = json.loads(extract_json_block(summary_response))
                    final_answer = clean_answer_text(summary_json.get("answer", ""))
                    suggested_actions = summary_json.get("suggested_actions", [])
                except Exception as e:
                    logger.error(f"[BoxAssistant] Failed to parse summary JSON: {e}")
                    final_answer = "Here are the results I found"

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

        except Exception as e:
            REQUEST_ERRORS.labels(endpoint="/api/ai/query", method="POST").inc()
            logger.error(f"[BoxAssistant] Query execution failed: {e}")
            return self._fallback_response(query_type)

        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="/api/ai/query", method="POST").observe(latency)

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