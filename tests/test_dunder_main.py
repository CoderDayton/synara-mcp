"""`python -m synara` dispatches to synara.main.main()."""

from __future__ import annotations

import runpy
from unittest.mock import patch


def test_module_invocation_calls_main() -> None:
    with patch("synara.main.main") as fake_main:
        fake_main.return_value = 0
        runpy.run_module("synara", run_name="__main__")
        fake_main.assert_called_once()
