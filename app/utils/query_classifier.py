import json
from difflib import SequenceMatcher
from typing import Optional, Dict, List
from sentence_transformers import SentenceTransformer, util
from app.services.schema_provider import RELEVANT_TABLES
from app.services.caching_service import CachingService

model: Optional[SentenceTransformer] = None
cache: Optional[CachingService] = None

CATEGORY_EXAMPLES = {
    "GENERAL": [
        "What is an electric vehicle?",
        "Compare electric and hybrid vehicles",
        "How do car engines work?",
        "Explain auction process"
    ],
    "VEHICLE_SEARCH": [
        "Show me SUVs under $30k",
        "Find electric cars in my area",
        "Search for used trucks",
        "List hybrid vehicles",
        "What is the average price of Toyota cars?"
    ],
    "AUCTION_SEARCH": [
        "What auctions are currently live?",
        "Show upcoming car auctions",
        "List active auctions",
        "Find auctions near me"
    ],
    "FINANCE_CALC": [
        "Calculate monthly payment for $20k car",
        "Estimate loan for vehicle",
        "Finance options for new car",
        "What will my EMI be?"
    ],
    "USER_SPECIFIC": [
        "What vehicles have I recently viewed?",
        "Did I win any auctions?",
        "Show my saved searches",
        "What bids have I placed?"
    ]
}

FORBIDDEN_KEYWORDS = [
    "drop", "delete", "alter", "insert", "update",
    "--", "exec", "truncate"
]

def is_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def extract_user_specific_tables(relevant_tables: Dict[str, List[str]]):
    return [
        table.lower()
        for table, columns in relevant_tables.items()
        if any("UserId" in col for col in columns)
    ]

def boost_user_specific_by_table_semantics(query_embedding, user_tables):
    table_embeddings = model.encode(user_tables, convert_to_tensor=True)
    similarity = util.cos_sim(query_embedding, table_embeddings).max().item()
    return similarity > 0.5

def is_query_unsafe(query: str, user_context: dict, relevant_tables: dict) -> bool:
    lowered_query = query.lower()

    if any(keyword in lowered_query for keyword in FORBIDDEN_KEYWORDS):
        return True

    if "reserve" in lowered_query and "price" in lowered_query:
        return True

    if user_context:
        if "user" in lowered_query and str(user_context["user_id"]) not in lowered_query:
            return True
        if user_context.get("name") and user_context["name"].lower() in lowered_query:
            if f"my {user_context['name'].lower()}" not in lowered_query and "me" not in lowered_query:
                return True
        if user_context.get("email") and user_context["email"].lower() in lowered_query:
            if f"my {user_context['email'].lower()}" not in lowered_query and "me" not in lowered_query:
                return True

    user_fields = relevant_tables.get("Users", [])
    sensitive_fields = [f.lower() for f in user_fields if f.lower() not in ["userid", "id"]]
    if any(field in lowered_query for field in sensitive_fields):
        return True
    

    return False

async def classify_query(query: str, user_context: dict = None) -> str:
    if is_query_unsafe(query, user_context, RELEVANT_TABLES):
        return "UNSAFE"
    cache_key = f"embedding:query:{query}"
    cached_embedding = await cache.redis.get(cache_key) if cache else None
    if cached_embedding:
        query_embedding = json.loads(cached_embedding)
    else:
        query_embedding = model.encode(query, convert_to_tensor=True)
        if cache:
            await cache.redis.setex(cache_key, 3600, json.dumps(query_embedding.tolist()))

    scores = {}
    for category, examples in CATEGORY_EXAMPLES.items():
        cat_key = f"embedding:category:{category}"
        cached_examples = await cache.redis.get(cat_key) if cache else None
        if cached_examples:
            example_embeddings = json.loads(cached_examples)
        else:
            example_embeddings = model.encode(examples, convert_to_tensor=True)
            if cache:
                await cache.redis.setex(cat_key, 86400, json.dumps(example_embeddings.tolist()))

        similarity = util.cos_sim(query_embedding, example_embeddings).max().item()
        scores[category] = similarity

    lowered_query = query.lower()

    definitional_triggers = ["what is", "explain", "define", "how do", "how does", "difference between"]
    if any(lowered_query.startswith(trigger) or trigger in lowered_query for trigger in definitional_triggers):
        scores["GENERAL"] += 0.15

    for table, cols in RELEVANT_TABLES.items():
        for col in cols:
            col_lower = col.lower()
            if col_lower in lowered_query:
                if table.lower() == "vehicles":
                    scores["VEHICLE_SEARCH"] += 0.1
                elif table.lower() == "auctions":
                    scores["AUCTION_SEARCH"] += 0.1

    user_tables = extract_user_specific_tables(RELEVANT_TABLES)
    if boost_user_specific_by_table_semantics(query_embedding, user_tables):
        scores["USER_SPECIFIC"] += 0.1

    return max(scores, key=scores.get)

async def preload_model_and_cache(redis_client):
    global model, cache
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    if cache is None:
        cache = CachingService(redis_client)
    print("Model and cache initialized")
