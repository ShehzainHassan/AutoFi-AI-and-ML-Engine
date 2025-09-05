import re
from typing import Any

class AnswerGenerator:
    @staticmethod
    def generate(data: Any, answer: str) -> str:
        if not data:
            return answer

        if isinstance(data, list):
            if len(data) == 1 and isinstance(data[0], dict):
                # Single row: summarize aggregates
                return AnswerGenerator._summarize_aggregates(data[0])
            elif 1 < len(data) <= 3 and all(isinstance(row, dict) for row in data):
                # Small number of rows: summarize each
                summaries = [AnswerGenerator._summarize_aggregates(row) for row in data]
                return " | ".join(summaries)
            else:
                # Large number of rows
                return f"Sure, here are {len(data)} results"

        if isinstance(data, dict):
            return AnswerGenerator._summarize_aggregates(data)

        return "Results retrieved successfully."

    @staticmethod
    def _summarize_aggregates(fields: dict) -> str:
        phrases = []
        for key, value in fields.items():
            label = AnswerGenerator._prettify_key(key)
            formatted = AnswerGenerator._format_value(value)
            phrases.append(f"The {label} is {formatted}")
        return " and ".join(phrases)

    @staticmethod
    def _prettify_key(key: str) -> str:
        key = key.replace("_", " ")
        key = re.sub(r'([a-z])([A-Z])', r'\1 \2', key)
        return key.lower().strip()

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{round(value, 2):,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)
