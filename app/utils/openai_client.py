import time
import logging
import asyncio
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
    Optimized async OpenAI client with concurrency limits, retries,
    metrics, and minimal overhead for low latency.
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
        model_to_use = model or self.model
        max_tokens_to_use = max_tokens or self.max_tokens
        temperature_to_use = temperature if temperature is not None else self.temperature

        delay = 0.5

        for attempt in range(1, max_attempts + 1):
            async with self.semaphore:
                start_time = time.perf_counter()
                try:
                    response_stream = await self.client.chat.completions.create(
                        model=model_to_use,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature_to_use,
                        max_tokens=max_tokens_to_use,
                        stream=True,
                        timeout=self.timeout,
                    )

                    chunks = []
                    async for chunk in response_stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            chunks.append(delta.content)

                    content = "".join(chunks)

                    latency = time.perf_counter() - start_time
                    OPENAI_LATENCY.observe(latency)
                    OPENAI_REQUESTS.labels(status="success").inc()
                    return content

                except AuthenticationError as e:
                    OPENAI_REQUESTS.labels(status="auth_error").inc()
                    logger.error(f"[OpenAI] Authentication failed: {e}")
                    return "Invalid OpenAI API key."

                except OpenAIError as e:
                    latency = time.perf_counter() - start_time
                    OPENAI_LATENCY.observe(latency)
                    OPENAI_REQUESTS.labels(status="failure").inc()

                    logger.warning(
                        f"[OpenAI] Attempt {attempt} failed: {type(e).__name__} | {str(e)}"
                    )

                    if attempt == max_attempts:
                        return "Failed to generate AI response after multiple attempts."

                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 2)