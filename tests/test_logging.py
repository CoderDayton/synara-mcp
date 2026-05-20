"""configure_logging: stderr handler, stdout strip, idempotent re-install."""

from __future__ import annotations

import io
import logging
import sys
from collections.abc import Iterator

import pytest

from synara.logging import _OWN_HANDLER_MARK, configure_logging


@pytest.fixture
def _restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    try:
        yield
    finally:
        root.handlers.clear()
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def _own_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, _OWN_HANDLER_MARK, False)]


@pytest.mark.usefixtures("_restore_root_logger")
def test_installs_single_stderr_handler() -> None:
    configure_logging("INFO")
    own = _own_handlers()
    assert len(own) == 1
    handler = own[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr


@pytest.mark.usefixtures("_restore_root_logger")
def test_sets_root_level() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


@pytest.mark.usefixtures("_restore_root_logger")
def test_idempotent_repeated_configuration() -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(_own_handlers()) == 1


@pytest.mark.usefixtures("_restore_root_logger")
def test_strips_stdout_handler() -> None:
    stray = logging.StreamHandler(stream=sys.stdout)
    logging.getLogger().addHandler(stray)
    configure_logging("INFO")
    stdout_handlers = [
        h for h in logging.getLogger().handlers if getattr(h, "stream", None) is sys.stdout
    ]
    assert stdout_handlers == []


@pytest.mark.usefixtures("_restore_root_logger")
def test_preserves_unrelated_stderr_handler() -> None:
    foreign_stream = io.StringIO()
    foreign = logging.StreamHandler(stream=foreign_stream)
    logging.getLogger().addHandler(foreign)
    configure_logging("INFO")
    assert foreign in logging.getLogger().handlers


@pytest.mark.usefixtures("_restore_root_logger")
def test_formatter_includes_timestamp_level_name() -> None:
    configure_logging("INFO")
    handler = _own_handlers()[0]
    fmt = handler.formatter
    assert fmt is not None
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    line = fmt.format(record)
    assert "INFO" in line
    assert "x" in line
    assert "hello" in line
