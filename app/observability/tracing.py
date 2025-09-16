from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def setup_tracing(app):
    """
    Setup OpenTelemetry tracing with OTLP exporter.
    Sends traces to Jaeger / OTEL Collector.
    """

    resource = Resource.create({"service.name": "boxcars-ai"})

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")

    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    FastAPIInstrumentor.instrument_app(app)
