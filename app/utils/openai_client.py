import time
import logging
from openai import OpenAI, OpenAIError, AuthenticationError

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call_openai_with_retry(self, prompt: str, max_attempts: int = 3) -> str:
        delay = 1
        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(
                    f"[OpenAI] Attempt {attempt} | model={self.model} | "
                    f"prompt_len={len(prompt)} | preview={prompt[:200]!r}"
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are BoxAssistant, an AI assistant for AutoFi."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=20,
                    temperature=0.2,
                    max_tokens=500
                )
                logger.debug(f"[OpenAI] Attempt {attempt} succeeded. Usage: {response.usage}")
                logger.debug(f"[OpenAI] Finish reason: {response.choices[0].finish_reason}")
                logger.debug(f"[OpenAI] Raw content: {response.choices[0].message.content!r}")
                return response.choices[0].message.content

            except AuthenticationError as e:
                logger.error(f"[OpenAI] Authentication failed: {e}")
                return "Invalid OpenAI API key. Please check your credentials"

            except OpenAIError as e:
                logger.warning(
                    f"[OpenAI] Attempt {attempt} failed. "
                    f"Type={type(e).__name__} | Message={str(e)}"
                )
                if hasattr(e, 'status_code'):
                    logger.warning(f"[OpenAI] HTTP status: {e.status_code}")
                if hasattr(e, 'request_id'):
                    logger.warning(f"[OpenAI] Request ID: {e.request_id}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.warning(f"[OpenAI] Response body: {e.response}")

                if attempt == max_attempts:
                    return "Failed to generate AI response after multiple attempts."
                time.sleep(delay)
                delay *= 2