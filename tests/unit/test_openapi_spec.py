from __future__ import annotations

from app.main import app


def test_openapi_endpoint_is_enabled() -> None:
    assert app.openapi_url == "/openapi.json"
    route_paths = {getattr(route, "path", "") for route in app.routes}
    assert "/openapi.json" in route_paths


def test_openapi_schema_includes_guardrails_paths() -> None:
    schema = app.openapi()
    paths = schema.get("paths", {})
    assert "/v1/guardrails/apply" in paths
    assert "/v1/guardrails/apply-stream" in paths
    assert "/v1/guardrails/capabilities" in paths
    assert "/v1/guardrails/sessions/{session_id}/finalize" in paths
