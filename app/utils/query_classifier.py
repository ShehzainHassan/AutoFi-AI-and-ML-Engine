import json
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from functools import lru_cache
from app.services.caching_service import CachingService
from rapidfuzz import fuzz

FORBIDDEN_KEYWORDS = [
    "drop", "delete", "alter", "insert", "update", "truncate", "--", "exec"
]

model: Optional[SentenceTransformer] = None
cache: Optional[CachingService] = None

class QueryClassifier:
    """Classify natural language queries into categories using embeddings"""

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_query_patterns()

    @lru_cache(maxsize=1)
    def _build_query_patterns(self):
        """Define query categories and embed their example prompts"""
        self.QUERY_PATTERNS: Dict[str, Dict] = {
            "GENERAL": {
                "examples": [
                    "What is an electric vehicle?",
                    "Compare electric and hybrid vehicles",
                    "Explain auction process",
                    "Define car transmission"
                ]
            },
            "VEHICLE_SEARCH": {
                "examples": [
                    "Show me SUVs under $30k",
                    "Find electric cars",
                    "List hybrid vehicles",
                    "What is the average price of Toyota cars?"
                ]
            },
            "AUCTION_SEARCH": {
                "examples": [
                    "What auctions are currently live?",
                    "Show upcoming car auctions",
                    "List active auctions",
                    "Find auctions near me"
                ]
            },
            "FINANCE_CALC": {
                "examples": [
                    "Calculate monthly payment for $20k car",
                    "Estimate loan for vehicle",
                    "Finance options for new car",
                    "What will my EMI be?"
                ]
            },
            "USER_SPECIFIC": {
                "examples": [
                    "What vehicles have I recently viewed?",
                    "Did I win any auctions?",
                    "Show my saved searches",
                    "What bids have I placed?"
                ]
            }
        }

        self.pattern_embeddings: Dict[str, torch.Tensor] = {}
        for category, info in self.QUERY_PATTERNS.items():
            self.pattern_embeddings[category] = self.model.encode(
                info["examples"], convert_to_tensor=True
            )

    def is_query_unsafe(self, query: str, user_context: dict) -> bool:
            """Check if query is unsafe due to SQL injection or sensitive data access"""

            q = query.lower()

            # 1. Block obvious SQL injection keywords (fuzzy tolerance)
            for keyword in FORBIDDEN_KEYWORDS:
                if fuzz.partial_ratio(keyword, q) > 85:  # allow typos like 'dropp' or 'deleet'
                    return True

            # 2. Block "reserve price" queries (with typo tolerance)
            if fuzz.partial_ratio("reserve price", q) > 80:
                return True

            # 3. User-specific info checks
            if user_context:
                user_id = str(user_context.get("user_id", "")).lower()
                user_email = str(user_context.get("email", "")).lower()
                user_name = str(user_context.get("name", "")).lower()

                # Possible sensitive terms
                sensitive_terms = ["user id", "userid", "user email", "email", "username", "user name"]

                for term in sensitive_terms:
                    if fuzz.partial_ratio(term, q) > 80:
                        # If user ID / email / name in query does not match logged-in user → UNSAFE
                        if user_id and user_id not in q \
                        and user_email and user_email not in q \
                        and user_name and user_name not in q:
                            return True

            return False

    async def classify(self, query: str, user_context: dict = None) -> Dict[str, any]:
        """Classify query into category and return confidence scores"""
        if self.is_query_unsafe(query, user_context):
            return {"category": "UNSAFE", "confidence_scores": {}}

        cache_key = f"embedding:query:{query}"
        cached = await cache.redis.get(cache_key) if cache else None
        if cached:
            query_embedding = torch.tensor(json.loads(cached))
        else:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            if cache:
                await cache.redis.setex(
                    cache_key, 3600, json.dumps(query_embedding.tolist())
                )

        scores: Dict[str, float] = {}
        for category, embeddings in self.pattern_embeddings.items():
            similarity = util.cos_sim(query_embedding, embeddings).max().item()
            scores[category] = similarity

        q = query.lower()
        definitional_triggers = ["what is", "explain", "define", "difference between"]
        if any(trigger in q for trigger in definitional_triggers):
            scores["GENERAL"] = scores.get("GENERAL", 0) + 0.15

        for category, val in scores.items():
            scores[category] = max(0.0, min(1.0, (val + 1) / 2))

        best_category = max(scores, key=scores.get)
        return {"category": best_category, "confidence_scores": scores}


query_classifier = QueryClassifier()

async def classify_query(query: str, user_context: dict = None) -> Dict[str, any]:
    return await query_classifier.classify(query, user_context)

async def preload_model_and_cache(redis_client):
    global model, cache
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    if cache is None:
        cache = CachingService(redis_client)
    print("Query classifier initialized")
