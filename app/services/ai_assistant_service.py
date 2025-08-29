import logging
from typing import Any, Dict, Optional, List
from openai import OpenAI, OpenAIError, AuthenticationError
from app.services.schema_provider import get_schema_context
from app.services.query_executor import QueryExecutor
import asyncio
from app.schemas.ai_schemas import AIResponseModel
from fastapi.encoders import jsonable_encoder
import simplejson as json
import time
from app.schemas.ai_schemas import UIType

logger = logging.getLogger(__name__)

class AIResponse:
    def __init__(self, data: Any, ui_type: UIType):
        self.data = data
        self.ui_type = ui_type

def classify_user_query(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["finance", "loan", "payment", "affordability"]):
        return "FINANCE_CALC"
    elif any(x in q for x in ["auction", "bid", "watchlist", "reserve", "current price"]):
        return "AUCTION_QUERY"
    elif any(x in q for x in ["vehicle", "car", "model", "make", "features"]):
        return "VEHICLE_SEARCH"
    elif any(x in q for x in ["my", "saved", "history", "auto bid"]):
        return "USER_SPECIFIC"
    else:
        return "GENERAL"

def extract_sources_from_response(response: str) -> List[str]:
    sources = []
    for line in response.splitlines():
        line = line.strip()
        if line.lower().startswith("source:"):
            sources.append(line.split(":", 1)[1].strip())
        elif line.lower().startswith("sources:"):
            raw = line.split(":", 1)[1].strip()
            if raw.startswith("[") and raw.endswith("]"):
                raw_list = raw[1:-1].split(",")
                sources.extend([item.strip() for item in raw_list])
            else:
                sources.append(raw)
    return sources

from datetime import datetime
def json_safe(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

def format_context_for_prompt(context: Optional[dict]) -> str:
    """Format user context into labeled sections + compact JSON fallback."""
    if not context:
        return ""

    parts = []
    dotnet_ctx = context.get("dotnet_context", {})
    ml_ctx = context.get("ml_context", {})

    # .NET Context Summary
    if dotnet_ctx:
        auction_history = dotnet_ctx.get("auction_history", [])
        auto_bids = dotnet_ctx.get("auto_bid_settings", [])
        saved_searches = dotnet_ctx.get("saved_searches", [])

    parts.append(
        f".NET Context Summary:\n"
        f"- Auctions participated: {len(auction_history)}\n"
        f"- Auto-bid configurations: {len(auto_bids)}\n"
        f"- Saved searches: {len(saved_searches)}\n"
        f"(Includes bidding history, auto-bid strategies, and personalized search preferences.)"
    )

    # ML Context Summary
    if ml_ctx:
        interactions = ml_ctx.get("interactions", [])
        analytics_events = ml_ctx.get("analytics_events", [])

    parts.append(
        f"ML Context Summary (last 5 entries only):\n"
        f"- Recent interactions: {len(interactions)}\n"
        f"- Recent behavioral events: {len(analytics_events)}\n"
        f"(These reflect the user's latest engagement patterns for personalization.)"
    )

    compact_json = json.dumps(context, separators=(",", ":"), default=json_safe)

    return "\n".join(parts) + "\nRaw Context (compact JSON, for reference only):\n" + compact_json

class AIQueryService:
    def __init__(self, api_key: str, query_executor: QueryExecutor, model: str = "gpt-3.5-turbo"):
        self.query_executor = query_executor
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.schema_context = get_schema_context()

    def get_suggested_actions(self, query_type: str, user_query: str) -> List[str]:
        q = user_query.lower()
        actions = []

        if query_type == "VEHICLE_SEARCH":
            if "compare" in q:
                actions = ["Compare selected vehicles", "View side-by-side specs"]
            elif "cheap" in q or "budget" in q:
                actions = ["Sort by price", "Filter by mileage"]
            else:
                actions = ["View vehicle details", "Save to favorites"]

        elif query_type == "AUCTION_QUERY":
            if "recommend" in q or "top" in q:
                actions = ["View recommended vehicles", "Sort by mileage"]
            elif "current price" in q:
                actions = ["Track price changes", "Set bid alert"]
            else:
                actions = ["View auction details", "Add to watchlist"]

        elif query_type == "FINANCE_CALC":
            actions = ["Adjust loan duration", "Change down payment", "View EMI breakdown"]

        elif query_type == "USER_SPECIFIC":
            if "watchlist" in q:
                actions = ["Remove from watchlist", "View auction history"]
            elif "autobid" in q:
                actions = ["Update autobid strategy", "Pause autobid"]
            else:
                actions = ["View saved searches", "Review preferences"]

        else:
            actions = ["Ask another question", "Explore vehicle categories"]

        return actions

    def _call_openai_with_retry(self, prompt: str, max_attempts: int = 3) -> str:
        delay = 1  
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"OpenAI attempt {attempt}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are BoxAssistant, an AI assistant for AutoFi."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except AuthenticationError as e:
                logger.error(f"Authentication failed: {e}")
                return "Invalid OpenAI API key. Please check your credentials"
            except OpenAIError as e:
                logger.warning(f"OpenAI attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    return "Failed to generate AI response after multiple attempts."
                time.sleep(delay)
                delay *= 2  
        
    def generate_response_sync(
        self,
        user_query: str,
        user_id: int = None,
        context: Optional[dict] = None
    ) -> AIResponseModel:
        """Generates AI response in a format fully aligned with AIResponseModel."""
        query_type = classify_user_query(user_query)
        suggested_actions = self.get_suggested_actions(query_type, user_query)

        # Base prompt
        prompt = f"""
        You are BoxAssistant, an AI assistant for AutoFi.
        Use the following database schema and car features context to answer the user query:
        {self.schema_context}

        User query: {user_query}
        """
        if context:
            prompt += "\n\nUser Context Summary:\n"
            prompt += format_context_for_prompt(context)

        prompt += "\nInstructions: Return a SQL query if data is needed, else provide a personalized explanation."

        raw_response = self._call_openai_with_retry(prompt)
        sources = extract_sources_from_response(raw_response)
        logger.info(f"OpenAI returned: {raw_response}")


        data: Any = raw_response
        ui_type: UIType = UIType.TEXT

        is_sql = raw_response.lower().startswith("select")
        if is_sql and self.query_executor.validate_query(raw_response):
            try:
                params: Dict[str, Any] = {}
                if user_id and query_type in ["USER_SPECIFIC", "AUCTION_QUERY", "FINANCE_CALC"]:
                    params["user_id"] = user_id
                rows = asyncio.run(self.query_executor.execute_safe_query(raw_response, params=params))
                data = rows

                if query_type == "VEHICLE_SEARCH":
                    ui_type = UIType.CARD_GRID
                elif query_type in ["AUCTION_QUERY", "USER_SPECIFIC"]:
                    ui_type = UIType.TABLE
                elif query_type == "FINANCE_CALC":
                    ui_type = UIType.CALCULATOR
            except Exception as e:
                logger.error(f"SQL execution failed: {e}")
                data = {"error": "Failed to fetch data from database."}
                ui_type = UIType.TEXT
        else:
            # UI detection via tags
            if "[TABLE]" in raw_response:
                ui_type = UIType.TABLE
                data = raw_response.replace("[TABLE]", "").strip()
            elif "[CARD_GRID]" in raw_response:
                ui_type = UIType.CARD_GRID
                data = raw_response.replace("[CARD_GRID]", "").strip()
            elif "[CALCULATOR]" in raw_response:
                ui_type = UIType.CALCULATOR
                data = raw_response.replace("[CALCULATOR]", "").strip()
            elif "[CHART]" in raw_response:
                ui_type = UIType.CHART
                data = raw_response.replace("[CHART]", "").strip()
    
        answer_str = json.dumps(data, indent=2, default=json_safe, use_decimal=True)
        return AIResponseModel(
            answer=answer_str,
            ui_type=ui_type,
            query_type=query_type,
            data = jsonable_encoder(data),
            suggested_actions=suggested_actions,
            sources=sources,
        )