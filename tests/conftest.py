# tests/conftest.py
import sys
import types
import pytest
from fastapi.testclient import TestClient

# ---- Create stub packages to satisfy imports at module import time ----
# phoenix
fake_phoenix = types.SimpleNamespace()
def _fake_launch_app():
    return types.SimpleNamespace(url=None)
fake_phoenix.launch_app = _fake_launch_app
sys.modules.setdefault("phoenix", fake_phoenix)

# phoenix.otel.register -> returns a tracer provider with get_tracer + context manager span
class _FakeSpan:
    def __enter__(self): return self
    def __exit__(self, *exc): pass
    def set_attribute(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass

class _FakeTracer:
    def start_as_current_span(self, *a, **k): return _FakeSpan()

class _FakeTracerProvider:
    def get_tracer(self, *_a, **_k): return _FakeTracer()

def _fake_register(project_name=None, endpoint=None):
    return _FakeTracerProvider()

fake_phoenix_otel = types.SimpleNamespace(register=_fake_register)
sys.modules.setdefault("phoenix.otel", fake_phoenix_otel)

# openinference.instrumentation.openai.OpenAIInstrumentor
class _FakeOpenAIInstrumentor:
    def instrument(self, *a, **k): pass
fake_oi_openai = types.SimpleNamespace(OpenAIInstrumentor=_FakeOpenAIInstrumentor)
sys.modules.setdefault("openinference.instrumentation.openai", fake_oi_openai)

# openinference.semconv.trace.SpanAttributes (only referenced symbol; provide placeholder)
fake_semconv_trace = types.SimpleNamespace(SpanAttributes=object)
sys.modules.setdefault("openinference.semconv.trace", fake_semconv_trace)

# openinference.instrumentation.TracerProvider (placeholder)
sys.modules.setdefault("openinference.instrumentation", types.SimpleNamespace(TracerProvider=object))

# opentelemetry.trace.StatusCode (just needs an attribute)
fake_otel_trace = types.SimpleNamespace(StatusCode=types.SimpleNamespace(OK="OK", ERROR="ERROR"))
sys.modules.setdefault("opentelemetry.trace", fake_otel_trace)

# ---- Now import the app module safely ----
import importlib
app_module = importlib.import_module("fastapi_phoenix_agent")

@pytest.fixture
def client():
    return TestClient(app_module.app)
