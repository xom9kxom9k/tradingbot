"""Docker artefacts exist and describe the two runtime services."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_dockerignore_keeps_secrets_and_data_out_of_the_context() -> None:
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for name in (".env", "data", "runs", "logs", ".git"):
        assert name in text


def test_dockerfile_is_multi_stage_and_non_root() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim AS builder" in text
    assert "FROM python:3.11-slim AS runtime" in text
    assert "USER app" in text
    assert "HEALTHCHECK" in text
    assert "uv pip install" in text


def test_compose_defines_bot_and_dashboard() -> None:
    payload = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = payload["services"]
    assert set(services) == {"bot", "dashboard"}
    assert services["bot"]["restart"] == "unless-stopped"
    assert services["dashboard"]["restart"] == "unless-stopped"
    ports = services["dashboard"]["ports"]
    assert any("8501" in str(item) for item in ports)
    bot_volumes = " ".join(str(item) for item in services["bot"]["volumes"])
    for mount in ("/app/data", "/app/runs", "/app/logs"):
        assert mount in bot_volumes
    env_file = services["bot"]["env_file"]
    assert ".env" in env_file if isinstance(env_file, list) else env_file == ".env"
