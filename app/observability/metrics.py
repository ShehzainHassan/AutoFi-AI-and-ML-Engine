from fastapi import Request
from prometheus_client import Counter, Histogram, start_http_server
import time

REQUEST_COUNT = Counter(
    "request_count_total", "Total number of requests", ["endpoint", "method", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["endpoint", "method"]
)
REQUEST_ERRORS = Counter(
    "request_errors_total", "Total number of failed AI requests", ["endpoint", "method"]
)

def attach_metrics(app):
    """
    Attach Prometheus metrics middleware to FastAPI app
    and expose metrics on port 8001.
    """
    start_http_server(8001)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        latency = time.time() - start_time

        endpoint = request.url.path
        method = request.method
        status_code = response.status_code

        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(latency)

        return response
