"""`python -m synara` dispatches to synara.main.main() and exits with its code."""

from __future__ import annotations

import runpy
from unittest.mock import patch

import pytest


def test_module_invocation_calls_main_and_exits() -> None:
    with patch("synara.main.main") as fake_main:
        fake_main.return_value = 0
        # __main__ wraps the return in ``raise SystemExit(main())`` so the
        # process exit status reflects main()'s return value.
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("synara", run_name="__main__")
        fake_main.assert_called_once()
        assert excinfo.value.code == 0


def test_module_invocation_propagates_nonzero_exit() -> None:
    with patch("synara.main.main") as fake_main:
        fake_main.return_value = 3
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("synara", run_name="__main__")
        assert excinfo.value.code == 3
