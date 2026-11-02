"""Regression checks for dependency, model-loading, and container hardening."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server
import synthid_score_server
import synthid_text_server


def test_optional_image_backends_pin_patched_runtimes() -> None:
    ctrlregen = (SCRIPTS / "requirements-ctrlregen.txt").read_text(encoding="utf-8")
    markdiffusion = (SCRIPTS / "requirements-markdiffusion.txt").read_text(encoding="utf-8")
    markdiffusion_pins = {
        line for line in markdiffusion.splitlines() if line and not line.startswith("#")
    }

    for requirements in (ctrlregen, markdiffusion):
        assert "torch==2.13.0.*" in requirements
        assert "setuptools==84.1.0" in requirements
        assert "diffusers==0.42.1" in requirements
        assert "transformers==5.22.1" in requirements
        assert "huggingface_hub==2.4.0" in requirements

    assert "torch>=2.4,<2.11" not in markdiffusion_pins
    assert "setuptools<81" not in markdiffusion_pins


def test_compose_applies_hardening_and_requires_authentication() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "WATERMARKS_SERVER_API_KEY:?set WATERMARKS_SERVER_API_KEY" in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:\n    - ALL" in compose
    assert compose.count("<<: *service-hardening") == 7


@pytest.mark.parametrize(
    ("module", "env_name", "argv"),
    (
        (server, "WATERMARKS_SERVER_API_KEY", ["server.py", "--host", "192.0.2.1"]),
        (
            synthid_text_server,
            "WATERMARKS_SYNTHID_TEXT_API_KEY",
            ["synthid_text_server.py", "--host", "192.0.2.1"],
        ),
        (
            synthid_score_server,
            "WATERMARKS_SYNTHID_SCORER_API_KEY",
            ["synthid_score_server.py", "--host", "192.0.2.1"],
        ),
    ),
)
def test_http_servers_refuse_unauthenticated_non_loopback_bind(
    module, env_name: str, argv: list[str], monkeypatch, capsys
) -> None:
    monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(module, "API_KEY", "")
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 2
    assert "API key is required" in capsys.readouterr().err
