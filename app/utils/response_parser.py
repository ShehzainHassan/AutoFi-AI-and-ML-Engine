import re
from typing import List, Tuple, Union
from app.schemas.ai_schemas import UIType

def detect_ui_type_from_response(response: str) -> UIType:
    lowered = response.lower()
    if "[table]" in lowered:
        return UIType.TABLE
    elif "[card_grid]" in lowered:
        return UIType.CARD_GRID
    elif "[calculator]" in lowered:
        return UIType.CALCULATOR
    elif "[chart]" in lowered:
        return UIType.CHART
    return UIType.TEXT

def clean_response_content(response: str) -> str:
    return re.sub(r"\[(table|card_grid|calculator|chart)\]", "", response, flags=re.IGNORECASE).strip()

def parse_ui_response(response: str) -> Tuple[UIType, Union[str, dict]]:
    ui_type = detect_ui_type_from_response(response)
    cleaned = clean_response_content(response)
    return ui_type, cleaned