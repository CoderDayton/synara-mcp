"""Settings + DashboardConfig env parsing and security posture."""

from __future__ import annotations

from pathlib import Path

import pytest

from synara.config import Settings, default_db_path
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


def test_bad_log_level_rejected_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in SYNARA_LOG_LEVEL must fail loudly at config load.

    The legacy code stored the raw string and crashed deep inside
    ``logging.setLevel`` after the server had already started; that left
    operators with an opaque traceback. The allowlist check rejects the
    misconfiguration up front.
    """
    monkeypatch.setenv("SYNARA_LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValueError, match="SYNARA_LOG_LEVEL"):
        Settings.from_env()


@pytest.mark.parametrize("level", ["debug", "INFO", "Warning", "ERROR"])
def test_valid_log_level_case_insensitive(monkeypatch: pytest.MonkeyPatch, level: str) -> None:
    monkeypatch.setenv("SYNARA_LOG_LEVEL", level)
    assert Settings.from_env().log_level == level.upper()


def test_excessive_batch_size_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 1M default ceiling was absurd; tighter per-var caps catch
    misconfig before the first request runs."""
    monkeypatch.setenv("SYNARA_EMBEDDING_BATCH_SIZE", "100000")
    with pytest.raises(ValueError, match="SYNARA_EMBEDDING_BATCH_SIZE"):
        Settings.from_env()


# --- embedding env validation -------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["nan", "abc", "0", "-5", "inf", "1e9"],
)
def test_embedding_timeout_invalid_rejected(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Non-numeric, non-finite, non-positive, or over-ceiling timeouts
    all raise at startup rather than failing on the first request."""
    monkeypatch.setenv("SYNARA_EMBEDDING_TIMEOUT", raw)
    with pytest.raises(ValueError, match="SYNARA_EMBEDDING_TIMEOUT"):
        Settings.from_env()


def test_embedding_timeout_valid_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNARA_EMBEDDING_TIMEOUT", "12.5")
    assert Settings.from_env().embedding.timeout_seconds == 12.5


@pytest.mark.parametrize("raw", ["0", "-1", "notanint", "200000"])
def test_embedding_dim_invalid_rejected(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("SYNARA_EMBEDDING_DIM", raw)
    with pytest.raises(ValueError, match="SYNARA_EMBEDDING_DIM"):
        Settings.from_env()


def test_embedding_dim_valid_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNARA_EMBEDDING_DIM", "512")
    assert Settings.from_env().embedding.dim == 512


@pytest.mark.parametrize("raw", ["0", "-3", "x", "40000"])
def test_embedding_max_seq_length_invalid_rejected(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("SYNARA_EMBEDDING_MAX_SEQ_LENGTH", raw)
    with pytest.raises(ValueError, match="SYNARA_EMBEDDING_MAX_SEQ_LENGTH"):
        Settings.from_env()


def test_embedding_max_seq_length_valid_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNARA_EMBEDDING_MAX_SEQ_LENGTH", "2048")
    assert Settings.from_env().embedding.max_seq_length == 2048


def test_trust_remote_code_defaults_true() -> None:
    assert Settings.from_env().embedding.trust_remote_code is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False)],
)
def test_trust_remote_code_env_parsed(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("SYNARA_EMBEDDING_TRUST_REMOTE_CODE", raw)
    assert Settings.from_env().embedding.trust_remote_code is expected


def test_trust_remote_code_bad_value_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNARA_EMBEDDING_TRUST_REMOTE_CODE", "maybe")
    with pytest.raises(ValueError, match="SYNARA_EMBEDDING_TRUST_REMOTE_CODE"):
        Settings.from_env()


def test_default_db_path_uses_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))
    assert default_db_path() == str(tmp_path / "synara-mcp" / "synara.db")


def test_default_db_path_falls_back_to_legacy_cache_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_home = tmp_path / "data"
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    legacy_db = cache_home / "synara-mcp" / "synara.db"
    legacy_db.parent.mkdir(parents=True)
    legacy_db.touch()

    assert default_db_path() == str(legacy_db)


def test_default_db_path_prefers_new_path_once_it_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_home = tmp_path / "data"
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    legacy_db = cache_home / "synara-mcp" / "synara.db"
    legacy_db.parent.mkdir(parents=True)
    legacy_db.touch()
    new_db = data_home / "synara-mcp" / "synara.db"
    new_db.parent.mkdir(parents=True)
    new_db.touch()

    assert default_db_path() == str(new_db)
