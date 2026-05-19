"""Settings + DashboardConfig env parsing and security posture."""

from __future__ import annotations

import pytest

from synara.config import Settings
from synara.features.dashboard.config import DashboardConfig

_DASH_VARS = (
    "SYNARA_DASHBOARD",
    "SYNARA_DASHBOARD_HOST",
    "SYNARA_DASHBOARD_PORT",
    "SYNARA_DASHBOARD_TOKEN",
)


@pytest.fixture(autouse=True)
def _clear_dashboard_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _DASH_VARS:
        monkeypatch.delenv(var, raising=False)


def test_default_disabled() -> None:
    cfg = DashboardConfig.from_env()
    assert cfg.enabled is False
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8765
    assert cfg.token is None
    assert cfg.is_loopback is True


def test_settings_holds_dashboard_default() -> None:
    settings = Settings.from_env()
    assert isinstance(settings.dashboard, DashboardConfig)
    assert settings.dashboard.enabled is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("1", True), ("ON", True), ("false", False), ("no", False)],
)
def test_bool_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("SYNARA_DASHBOARD", raw)
    assert DashboardConfig.from_env().enabled is expected


def test_bad_bool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNARA_DASHBOARD", "maybe")
    with pytest.raises(ValueError, match="must be a boolean"):
        DashboardConfig.from_env()


@pytest.mark.parametrize("bad", ["0", "70000", "abc", "-1"])
def test_bad_port_raises(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("SYNARA_DASHBOARD_PORT", bad)
    with pytest.raises(ValueError, match=r"must be (an integer|in 1\.\.)"):
        DashboardConfig.from_env()


def test_loopback_variants_allowed_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for host in ("127.0.0.1", "localhost", "::1"):
        monkeypatch.setenv("SYNARA_DASHBOARD", "true")
        monkeypatch.setenv("SYNARA_DASHBOARD_HOST", host)
        cfg = DashboardConfig.from_env()
        assert cfg.enabled is True
        assert cfg.is_loopback is True


def test_non_loopback_with_token_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNARA_DASHBOARD", "true")
    monkeypatch.setenv("SYNARA_DASHBOARD_HOST", "0.0.0.0")
    monkeypatch.setenv("SYNARA_DASHBOARD_TOKEN", "secret")
    cfg = DashboardConfig.from_env()
    assert cfg.is_loopback is False
    assert cfg.token == "secret"


def test_non_loopback_without_token_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNARA_DASHBOARD", "true")
    monkeypatch.setenv("SYNARA_DASHBOARD_HOST", "192.168.1.10")
    with pytest.raises(ValueError, match="without a token"):
        DashboardConfig.from_env()


def test_disabled_non_loopback_no_token_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Gate is off -> posture check must not fire.
    monkeypatch.setenv("SYNARA_DASHBOARD", "false")
    monkeypatch.setenv("SYNARA_DASHBOARD_HOST", "0.0.0.0")
    cfg = DashboardConfig.from_env()
    assert cfg.enabled is False


def test_token_redacted_in_repr() -> None:
    cfg = DashboardConfig(enabled=True, token="topsecret")
    assert "topsecret" not in repr(cfg)
