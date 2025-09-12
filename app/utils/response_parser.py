import re

def extract_json_block(text: str) -> str:
    text = re.sub(r"```(?:json|sql)?", "", text)
    text = text.replace('\\"', '"')
    text = text.replace("\\", "")
    return text.strip()


@staticmethod
def clean_answer_text(answer: str) -> str:
    """Remove markdown formatting or prefixes from AI answer."""
    import re
    cleaned = re.sub(r"\*+", "", answer)
    cleaned = re.sub(r"^\s*answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()