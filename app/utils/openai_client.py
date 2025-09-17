import time
import logging
import asyncio
from typing import Tuple, Union
from openai import AsyncOpenAI, OpenAIError, AuthenticationError
from config.app_config import settings
from prometheus_client import Counter, Histogram

OPENAI_REQUESTS = Counter(
    "openai_requests_total",
    "Total number of OpenAI API requests",
    ["status"]
)
OPENAI_LATENCY = Histogram(
    "openai_request_latency_seconds",
    "Latency of OpenAI API requests in seconds"
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Async OpenAI client with concurrency limits, queuing, retries,
    metrics, and performance monitoring.
    """

    def __init__(self, max_concurrent_requests: int = 5):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.timeout = int(settings.OPENAI_TIMEOUT)

        self.client = AsyncOpenAI(api_key=self.api_key)

        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def call_openai_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        return_usage: bool = False
    ) -> Union[str, Tuple[str, dict]]:
        """
        Call OpenAI API with retry, queuing, and performance monitoring.
        """

        delay = 1
        for attempt in range(1, max_attempts + 1):
            async with self.semaphore:  
                start_time = time.perf_counter()
                try:
                    logger.debug(
                        f"[OpenAI] Attempt {attempt} | model={self.model} | "
                        f"prompt_len={len(prompt)} | preview={prompt[:200]!r}"
                    )

                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are BoxAssistant, an AI assistant for AutoFi."},
                            {"role": "user", "content": prompt}
                        ],
                        timeout=self.timeout,
                        temperature=0.2,
                        max_tokens=self.max_tokens,
                    )

                    latency = time.perf_counter() - start_time
                    OPENAI_LATENCY.observe(latency)
                    OPENAI_REQUESTS.labels(status="success").inc()

                    logger.debug(
                        f"[OpenAI] Success in {latency:.2f}s | Usage: {response.usage} | "
                        f"Finish reason: {response.choices[0].finish_reason}"
                    )
                    logger.debug(f"[OpenAI] Raw content: {response.choices[0].message.content!r}")

                    if return_usage:
                        return response.choices[0].message.content, response.usage
                    return response.choices[0].message.content

                except AuthenticationError as e:
                    OPENAI_REQUESTS.labels(status="auth_error").inc()
                    logger.error(f"[OpenAI] Authentication failed: {e}")
                    return "Invalid OpenAI API key. Please check your credentials"

                except OpenAIError as e:
                    latency = time.perf_counter() - start_time
                    OPENAI_LATENCY.observe(latency)
                    OPENAI_REQUESTS.labels(status="failure").inc()

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

                    await asyncio.sleep(delay)
                    delay *= 2