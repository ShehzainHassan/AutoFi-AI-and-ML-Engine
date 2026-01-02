from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import os
import logging

logger = logging.getLogger(__name__)


def setup_tracing(app):
    """
    Setup OpenTelemetry tracing with OTLP exporter.
    Sends traces to OTEL Collector if available, otherwise uses console exporter.
    """
    try:
        resource = Resource.create({"service.name": "boxcars-ai"})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # Check if OTEL endpoint is configured via environment variable
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        
        # Only use OTLP exporter if endpoint is explicitly set and not localhost
        # (localhost won't work in Railway - it will cause connection errors)
        if otlp_endpoint and "localhost" not in otlp_endpoint.lower():
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(span_processor)
                logger.info(f"OpenTelemetry tracing enabled with OTLP exporter: {otlp_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to setup OTLP exporter, using console exporter: {e}")
                # Fallback to console exporter
                console_exporter = ConsoleSpanExporter()
                span_processor = BatchSpanProcessor(console_exporter)
                provider.add_span_processor(span_processor)
        else:
            # Use console exporter if no valid endpoint configured (default for Railway)
            logger.info("OpenTelemetry tracing using console exporter (no OTLP endpoint configured)")
            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)
            provider.add_span_processor(span_processor)

        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry tracing: {e}")
        logger.warning("Continuing without tracing - application will still work")
