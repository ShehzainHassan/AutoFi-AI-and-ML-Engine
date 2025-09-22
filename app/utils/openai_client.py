import time
import logging
import asyncio
from typing import Union, Tuple
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
    metrics, and performance monitoring. Streaming is always enabled.
    """

    def __init__(self, max_concurrent_requests: int = 5):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.timeout = int(settings.OPENAI_TIMEOUT)
        self.temperature = settings.OPENAI_TEMPERATURE

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def call_openai_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None
    ) -> str:
        delay = 1
        model_to_use = model or self.model
        max_tokens_to_use = max_tokens or self.max_tokens
        temperature_to_use = temperature if temperature is not None else self.temperature

        for attempt in range(1, max_attempts + 1):
            async with self.semaphore:
                start_time = time.perf_counter()
                try:
                    logger.debug(
                        f"[OpenAI] Attempt {attempt} | model={model_to_use} | "
                        f"prompt_len={len(prompt)} | preview={prompt[:200]!r}"
                    )

                    response_stream = await self.client.chat.completions.create(
                        model=model_to_use,
                        messages=[
                            {"role": "system", "content": "You are BoxAssistant, an AI assistant for AutoFi."},
                            {"role": "user", "content": prompt}
                        ],
                        timeout=self.timeout,
                        temperature=temperature_to_use,
                        max_tokens=max_tokens_to_use,
                        stream=True
                    )

                    content = ""
                    async for chunk in response_stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content += delta.content

                    latency = time.perf_counter() - start_time
                    OPENAI_LATENCY.observe(latency)
                    OPENAI_REQUESTS.labels(status="success").inc()
                    return content

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