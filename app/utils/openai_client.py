import time
import logging
from openai import OpenAI, OpenAIError, AuthenticationError
from config.app_config import settings  

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.timeout = int(settings.OPENAI_TIMEOUT) 
        self.client = OpenAI(api_key=self.api_key)

    def call_openai_with_retry(self, prompt: str, max_attempts: int = 3, return_usage: bool = False) -> str:
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
                    timeout=self.timeout,
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                )

                logger.debug(f"[OpenAI] Attempt {attempt} succeeded. Usage: {response.usage}")
                logger.debug(f"[OpenAI] Finish reason: {response.choices[0].finish_reason}")
                logger.debug(f"[OpenAI] Raw content: {response.choices[0].message.content!r}")
                
                if return_usage:
                    return response.choices[0].message.content, response.usage
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
