import logging
import simplejson as json
from fastapi.encoders import jsonable_encoder
from opentelemetry import trace
from app.services.query_executor import QueryExecutor
from app.schemas.ai_schemas import AIResponseModel, UIType
from app.utils.context_builder import build_optimized_context
from app.utils.openai_client import OpenAIClient
from app.utils.ui_block_builder import UIBlockBuilder
from app.utils.query_classifier import classify_query
from app.utils.assistant_prompts import UNIFIED_PROMPT
import time

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("boxcars-ai")

class AIQueryService:
    def __init__(self, openai_client: OpenAIClient, query_executor: QueryExecutor):
        self.query_executor = query_executor
        self.openai_client = openai_client

    async def generate_response(self, user_query: str, user_id: int, context: dict = None) -> AIResponseModel:
        start_total = time.perf_counter()
        
        start_classify = time.perf_counter()
        result = await classify_query(user_query)
        elapsed_classify = time.perf_counter() - start_classify
        print(f"classify_query took {elapsed_classify:.4f}s")
        
        query_type = result["category"]

        if query_type == "UNSAFE":
            return self._fallback_response("UNSAFE")
        
        # Build optimized context
        start_context = time.perf_counter()
        prompt_context = build_optimized_context(query_type, user_query, user_id, context)
        prompt_context["user_query"] = user_query
        elapsed_context = time.perf_counter() - start_context
        print(f"build_optimized_context took {elapsed_context:.4f}s")

        # Single OpenAI call with unified prompt
        start_openai = time.perf_counter()
        with tracer.start_as_current_span("unified_openai_call"):
            prompt = UNIFIED_PROMPT.format(**prompt_context)
            print(prompt)
            raw_response = await self.openai_client.call_openai_with_retry(prompt)
        elapsed_openai = time.perf_counter() - start_openai
        print(f"OpenAI call took {elapsed_openai:.4f}s")
        
        start_parse = time.perf_counter()
        parsed = json.loads(raw_response)
        elapsed_parse = time.perf_counter() - start_parse
        print(f"Parsing OpenAI response took {elapsed_parse:.4f}s")
        
        try:
            sql_query = parsed.get("sql")
            data = []

            if sql_query and query_type not in {"GENERAL", "FINANCE_CALC"}:
                start_sql = time.perf_counter()
                user_context = {
                    "user_id": user_id,
                    "name": context.get("name") if context else None,
                    "email": context.get("email") if context else None,
                }
                data = await self.query_executor.execute_safe_query(sql_query, user_context)
                elapsed_sql = time.perf_counter() - start_sql
                print(f"SQL query execution took {elapsed_sql:.4f}s")

                if isinstance(data, list) and data and "error" in data[0]:
                    return self._fallback_response(query_type)

                if isinstance(data, list) and not data:
                    final_answer = "I could not find any results for your query."
                else:
                    final_answer = parsed.get("answer", "")

                start_ui = time.perf_counter()
                ui_type = UIType(parsed.get("ui_type", "TEXT").upper())
                chart_type = parsed.get("chart_type") if ui_type == UIType.CHART else None
                ui_block = UIBlockBuilder.build(
                    ui_type.value,
                    data or parsed.get("data_preview", {}),
                    final_answer,
                    chart_type=chart_type
                )
                elapsed_ui = time.perf_counter() - start_ui
                print(f"UIBlockBuilder took {elapsed_ui:.4f}s")

                return AIResponseModel(
                    answer=final_answer,
                    ui_type=ui_type,
                    query_type=query_type,
                    data=jsonable_encoder(data),
                    suggested_actions=parsed.get("suggested_actions", []),
                    sources=parsed.get("sources", []),
                    ui_block=ui_block,
                    chart_type=chart_type
                )
            else:
                final_answer = parsed.get("answer", "")
                ui_type = UIType(parsed.get("ui_type", "TEXT").upper())
                chart_type = parsed.get("chart_type") if ui_type == UIType.CHART else None

                start_ui2 = time.perf_counter()
                ui_block = UIBlockBuilder.build(
                    ui_type.value,
                    parsed.get("data_preview", {}),
                    final_answer,
                    chart_type=chart_type
                )
                elapsed_ui2 = time.perf_counter() - start_ui2
                print(f"UIBlockBuilder (no SQL) took {elapsed_ui2:.4f}s")

                return AIResponseModel(
                    answer=final_answer,
                    ui_type=ui_type,
                    query_type=query_type,
                    data=jsonable_encoder(parsed.get("data_preview", {})),
                    suggested_actions=parsed.get("suggested_actions", []),
                    sources=parsed.get("sources", []),
                    ui_block=ui_block,
                    chart_type=chart_type
                )                    

        except Exception as e:
            logger.error(f"[BoxAssistant] Unified response parsing failed: {e}")
            return self._fallback_response(query_type)
        finally:
            elapsed_total = time.perf_counter() - start_total
            print(f"Total generate_response took {elapsed_total:.4f}s")

    def _validate_unified_response(self, parsed: dict, query_type: str) -> bool:
        """Validate unified response has required fields"""
        has_answer = parsed.get("answer") and len(parsed["answer"]) > 10
        has_ui_type = parsed.get("ui_type") in ["TEXT", "TABLE", "CARD_GRID", "CALCULATOR", "CHART"]

        if query_type in {"GENERAL", "FINANCE_CALC"}:
            return has_answer and has_ui_type and parsed.get("sql") is None
        else:
            has_sql_or_data = parsed.get("sql") or parsed.get("data_preview")
            return has_answer and has_ui_type and has_sql_or_data
    
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