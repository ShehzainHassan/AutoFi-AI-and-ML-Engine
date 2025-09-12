from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from enum import Enum

class UIType(str, Enum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    CARD_GRID = "CARD_GRID"
    CALCULATOR = "CALCULATOR"
    CHART = "CHART"

class AIQueryRequest(BaseModel):
    user_id: Optional[int] = None
    question: str
    context: Optional[str] = None
    session_id: Optional[str] = None

class AIResponseModel(BaseModel):
    answer: str
    ui_type: UIType = UIType.TEXT
    query_type: str = "GENERAL"
    data: Any
    suggested_actions: Optional[List[str]] = []
    sources: Optional[List[str]] = [],
    ui_block: Optional[str] = None
    chart_type: Optional[str] = None


class UserContext(BaseModel):
    recent_searches: Optional[List[str]] = []
    auction_history: Optional[List[int]] = []
    favorites: Optional[List[int]] = []
    preferences: Optional[dict] = {}

class AIQueryFeedback(BaseModel):
    user_id: int
    session_id: Optional[str] = None
    query: str
    feedback: str

class EnrichedAIQuery(BaseModel):
    query: AIQueryRequest
    context: Dict[str, Any]

class FeedbackEnum(str, Enum):
    NOTVOTED = "NOTVOTED"
    UPVOTED = "UPVOTED"
    DOWNVOTED = "DOWNVOTED"

class AIQueryFeedback(BaseModel):
    message_id: int
    vote: FeedbackEnum

class PopularQueryDTO(BaseModel):
    text: str
    count: int
    last_asked: Optional[str]