"""HTTP-level CORS tests: preflight and simple responses.

Complements the configuration-level test in test_main.py — here we exercise
the real Starlette CORSMiddleware through TestClient.

Observations pinned here (verified via diagnostic on 2026-11-07):
- preflight with allowed origin+method+headers → 200 with allow-origin.
- preflight with allowed origin but bad method/header → 400, allow-origin
  still present (origin is allowed; only the operation is rejected).
- preflight with disallowed origin → 400, no allow-origin header.
- simple GET with allowed origin → 200 with allow-origin and vary: Origin.
- simple GET with disallowed origin → 200 without allow-origin.
- simple GET without Origin → 200 without CORS headers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.main import app

ALLOWED_ORIGINS = ("https://script.google.com", "https://docs.google.com")
SCRIPT_ORIGIN = "https://script.google.com"

DISALLOWED_ORIGINS = [
    "https://evil.example",
    "https://docs.google.com.attacker.example",
    "https://attacker.example/https://docs.google.com",
    "http://docs.google.com",
    "https://docs.google.com:444",
    "null",
    "https://docs.google.com.",
    "https://docs.google.com:443",
    "https://Docs.Google.Com",
    "https://docs.google.co",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _split_csv(value: str) -> list[str]:
    return [h.strip().lower() for h in value.split(",") if h.strip()]


def _vary_values(headers) -> list[str]:
    return _split_csv(headers.get("vary", ""))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def test_preflight_allowed_origin_returns_full_cors_headers(client: TestClient) -> None:
    response = client.options(
        "/api/edit",
        headers={
            "Origin": SCRIPT_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,X-API-Key",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == SCRIPT_ORIGIN
    assert response.headers.get("access-control-allow-credentials") == "true"
    assert "post" in _split_csv(response.headers.get("access-control-allow-methods", ""))
    allow_headers = _split_csv(response.headers.get("access-control-allow-headers", ""))
    assert "content-type" in allow_headers
    assert "x-api-key" in allow_headers
    assert "origin" in _vary_values(response.headers)


def test_preflight_disallowed_method_keeps_origin_but_rejects(client: TestClient) -> None:
    response = client.options(
        "/api/edit",
        headers={
            "Origin": SCRIPT_ORIGIN,
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert response.status_code == 400
    assert response.headers.get("access-control-allow-origin") == SCRIPT_ORIGIN
    assert "delete" not in _split_csv(response.headers.get("access-control-allow-methods", ""))


def test_preflight_disallowed_header_keeps_origin_but_rejects(client: TestClient) -> None:
    response = client.options(
        "/api/edit",
        headers={
            "Origin": SCRIPT_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Unapproved",
        },
    )
    assert response.status_code == 400
    assert response.headers.get("access-control-allow-origin") == SCRIPT_ORIGIN


@pytest.mark.parametrize("origin", DISALLOWED_ORIGINS)
def test_preflight_disallowed_origin_has_no_allow_origin(client: TestClient, origin: str) -> None:
    response = client.options(
        "/api/edit",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


# ---------------------------------------------------------------------------
# Simple requests
# ---------------------------------------------------------------------------


def test_simple_request_allowed_origin_gets_allow_origin(client: TestClient) -> None:
    response = client.get("/openapi.json", headers={"Origin": SCRIPT_ORIGIN})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == SCRIPT_ORIGIN
    assert "origin" in _vary_values(response.headers)


@pytest.mark.parametrize("origin", DISALLOWED_ORIGINS)
def test_simple_request_disallowed_origin_has_no_allow_origin(
    client: TestClient, origin: str
) -> None:
    response = client.get("/openapi.json", headers={"Origin": origin})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_simple_request_without_origin_has_no_cors_headers(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


# ---------------------------------------------------------------------------
# ENV cannot expand the allowlist (subprocess, isolated per scenario)
# ---------------------------------------------------------------------------

_PROBE = (
    "import json\n"
    "from src.main import _ALLOWED_GOOGLE_ORIGINS, _CORS_ORIGINS\n"
    "print(json.dumps({\n"
    "    'allowed': sorted(_ALLOWED_GOOGLE_ORIGINS),\n"
    "    'cors': sorted(_CORS_ORIGINS),\n"
    "}))\n"
)


def _run_probe(env_overrides: dict[str, str]) -> dict[str, list[str]]:
    env = {**os.environ, "PYTEST_RUNNING": "true", "ENV": "development"}
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=90,
        cwd=_PROJECT_ROOT,
        env=env,
        shell=False,
    )
    assert result.returncode == 0, f"probe failed: {result.stderr}"
    return json.loads(result.stdout)


_EXPECTED = sorted(ALLOWED_ORIGINS)


@pytest.mark.parametrize(
    "env_override",
    [
        {},
        {"CORS_ALLOWED_ORIGINS": "https://evil.example"},
        {"CORS_ALLOWED_ORIGINS": "https://script.google.com,https://evil.example"},
        {"CORS_ALLOWED_ORIGINS": "*"},
    ],
    ids=["env_absent", "env_foreign", "env_mixed", "env_wildcard"],
)
def test_cors_env_does_not_expand_allowlist(env_override: dict[str, str]) -> None:
    data = _run_probe(env_override)
    assert data["allowed"] == _EXPECTED
    assert data["cors"] == _EXPECTED
