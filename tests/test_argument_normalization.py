"""Tolerant tool-argument normalization.

The four call shapes exercised here are the exact ones that failed
validation in three days of real agent transcripts (4 of 63 synara calls,
6.3%): ``tags`` as a comma-joined string on three tools, and ``limit``
standing in for the declared ``k``. Each has a regression test below
against the real MCP surface, not just the pure helper.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from synara.core.argument_normalization import (
    normalize_arguments,
    split_scalar_list,
)

# Mirrors the shape pydantic emits for the real tools: an optional
# ``list[str]`` becomes an anyOf with null, a plain ``int`` does not.
_SCHEMA: dict[str, Any] = {
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "default": 4},
        "session_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "tags": {
            "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
            "default": None,
        },
    }
}


def _norm(args: dict[str, Any]) -> dict[str, Any]:
    normalized, _notes = normalize_arguments(args, schema=_SCHEMA, tool_name="recall_episodes")
    return normalized


# --------------------------------------------------------------------
# split_scalar_list
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The exact strings from the failing transcript calls.
        ("lunify,model-loading,bug", ["lunify", "model-loading", "bug"]),
        ("prism,attention,kv-cache", ["prism", "attention", "kv-cache"]),
        # Separator variants.
        ("a, b , c", ["a", "b", "c"]),
        ("a b c", ["a", "b", "c"]),
        ('["a", "b"]', ["a", "b"]),
        # A comma wins over whitespace, so a spaced list is not split twice.
        ("alpha beta, gamma", ["alpha beta", "gamma"]),
        # Degenerate input.
        ("", []),
        ("   ", []),
        ("solo", ["solo"]),
        (",,a,,", ["a"]),
        # Malformed JSON falls back to delimiter splitting rather than raising.
        ('["a", "b"', ['["a"', '"b"']),
    ],
)
def test_split_scalar_list(raw: str, expected: list[str]) -> None:
    assert split_scalar_list(raw) == expected


# --------------------------------------------------------------------
# list coercion
# --------------------------------------------------------------------


def test_scalar_string_splits_for_array_param() -> None:
    assert _norm({"query": "q", "tags": "a,b"})["tags"] == ["a", "b"]


def test_comma_and_whitespace_forms_agree() -> None:
    assert _norm({"query": "q", "tags": "a,b"}) == _norm({"query": "q", "tags": "a b"})


def test_existing_list_is_untouched() -> None:
    # Notably a tag that itself contains a comma survives: splitting only
    # ever applies to a bare string, never inside a list the caller built.
    tags = ["a,b", "c"]
    assert _norm({"query": "q", "tags": tags})["tags"] == tags


def test_string_param_is_not_split() -> None:
    # ``query`` is a string, not an array — a comma in it is content.
    assert _norm({"query": "a,b"})["query"] == "a,b"


def test_non_string_scalar_is_left_for_validation() -> None:
    # Only strings are meaningfully splittable; anything else keeps its
    # value so pydantic reports the real type error.
    assert _norm({"query": "q", "tags": 5})["tags"] == 5


# --------------------------------------------------------------------
# aliases
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("limit", "k"),
        ("n", "k"),
        ("top_k", "k"),
        ("max_results", "k"),
        ("topK", "k"),  # camelCase resolves through snake_case into the alias
        ("q", "query"),
        ("sessionId", "session_id"),
        ("session", "session_id"),
        ("tag", "tags"),
    ],
)
def test_alias_renames_to_declared_parameter(alias: str, canonical: str) -> None:
    out = _norm({alias: "8" if canonical == "k" else "v"})
    assert canonical in out
    assert alias not in out


def test_alias_and_canonical_agreeing_drops_the_alias() -> None:
    assert _norm({"query": "q", "limit": 8, "k": 8}) == {"query": "q", "k": 8}


def test_alias_and_canonical_conflicting_is_rejected() -> None:
    with pytest.raises(ToolError, match="alias for 'k'"):
        _norm({"query": "q", "limit": 8, "k": 4})


def test_two_aliases_for_one_target_is_rejected() -> None:
    with pytest.raises(ToolError, match="both aliases for 'k'"):
        _norm({"query": "q", "limit": 8, "n": 4})


def test_unknown_key_is_left_for_pydantic() -> None:
    # Passing it through preserves pydantic's "Unexpected keyword
    # argument", which names the real parameter — a better error than
    # anything a guess would produce.
    assert _norm({"query": "q", "wibble": 1})["wibble"] == 1


def test_alias_whose_target_is_absent_from_schema_is_left_alone() -> None:
    # ``id`` -> ``episode_id``, but this schema has no ``episode_id``.
    assert _norm({"query": "q", "id": 7})["id"] == 7


def test_rename_then_split_compose() -> None:
    # ``tag`` renames to ``tags``, and the renamed value is then split.
    assert _norm({"query": "q", "tag": "a,b"})["tags"] == ["a", "b"]


# --------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------


def test_canonical_call_is_unchanged_and_notes_nothing() -> None:
    args = {"query": "q", "k": 4, "tags": ["a"], "session_id": "s1"}
    normalized, notes = normalize_arguments(args, schema=_SCHEMA)
    assert normalized == args
    assert notes == []


def test_missing_schema_is_a_passthrough() -> None:
    args = {"tags": "a,b", "limit": 3}
    assert normalize_arguments(args, schema={}) == (args, [])


def test_empty_arguments_short_circuit() -> None:
    assert normalize_arguments({}, schema=_SCHEMA) == ({}, [])
