"""Tolerant argument normalization for the MCP tool surface.

Why this exists (measured, not speculative)
-------------------------------------------
Across three days of real agent transcripts, 4 of 63 ``synara`` tool
calls (6.3%) failed pydantic validation before reaching a handler, and
every one was a *shape* error rather than a semantic one:

* ``tags="a,b,c"`` supplied where ``list[str]`` is declared  (x3)
* ``limit=8`` supplied where the parameter is named ``k``    (x1)

Both parameters are documented correctly in the tool descriptions, so
documentation alone did not prevent the fault. Each failure cost a full
round-trip and returned a pydantic traceback instead of a result.

Design
------
Normalization runs as FastMCP middleware, which puts it *upstream* of
pydantic validation (``FastMCP._mcp_call_tool`` runs the middleware
chain, then re-enters ``call_tool`` with ``context.message.arguments``,
and only ``tool._run`` validates). That ordering is the whole point: the
declared signature stays narrow — ``tags: list[str]``, ``k: int`` — so
the published schema keeps teaching the canonical call, while a
near-miss costs a rename instead of a wasted turn.

Every rule is driven by the target tool's own JSON schema, never by a
hand-maintained per-tool table:

* an alias is applied only when the canonical parameter exists in that
  tool's schema and the caller did not already supply it;
* a scalar is split into a list only when the schema declares that
  parameter as an array of strings.

So new tools inherit the behaviour with no registration step, and a
rename can never shadow a real parameter.

Deliberately not handled
------------------------
String-to-number coercion (``k="8"``, ``salience="0.7"``). Pydantic's
lax mode already accepts those, so a rule for them would be dead code —
verified against the live tool surface, not assumed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

_LOG = logging.getLogger(__name__)

# Synonyms an agent reaches for when it does not recall the declared
# parameter name. Each maps to an ordered tuple of candidate targets; the
# first candidate that exists in the called tool's schema wins, so one
# synonym can serve tools that spell the same concept differently
# (``text`` means ``query`` on a recall tool and ``content`` on a store
# tool). Kept short and evidence-led rather than exhaustive: a synonym
# that is never emitted is dead weight, and one that guesses wrong is
# worse than a clear validation error.
_ALIASES: dict[str, tuple[str, ...]] = {
    # Result-count synonyms. ``limit`` is the one observed in transcripts;
    # the rest are the standard neighbours in the same semantic cluster.
    "limit": ("k",),
    "n": ("k",),
    "top_k": ("k",),
    "num_results": ("k",),
    "max_results": ("k",),
    "count": ("k",),
    # Search-text synonyms.
    "q": ("query",),
    "text": ("query", "content"),
    "search": ("query",),
    # Payload synonyms.
    "body": ("content",),
    "memory": ("content",),
    # Identifier synonyms.
    "id": ("episode_id",),
    # Namespace synonyms.
    "session": ("session_id",),
    "namespace": ("session_id",),
    # Tag synonyms.
    "tag": ("tags",),
    "labels": ("tags",),
}

# ``sessionId`` -> ``session_id``. Applied generically so every camelCase
# spelling of every parameter is covered without enumerating them.
_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _snake_case(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _candidates(key: str) -> tuple[str, ...]:
    """Ordered canonical names ``key`` might stand for, most direct first.

    Chains the two rules so ``topK`` resolves through its snake_case form
    ``top_k`` into the alias target ``k``.
    """
    out: list[str] = []
    for candidate in (
        *_ALIASES.get(key, ()),
        _snake_case(key),
        *_ALIASES.get(_snake_case(key), ()),
    ):
        if candidate != key and candidate not in out:
            out.append(candidate)
    return tuple(out)


def _accepts_string_array(prop: object) -> bool:
    """True when a JSON-schema property accepts an array of strings.

    Walks ``anyOf``/``oneOf`` so an optional ``list[str] | None`` — which
    pydantic renders as a union with ``null`` — is recognised. Unresolved
    ``$ref`` schemas deliberately return False: without the definition we
    cannot know the shape, and a wrong split is worse than no split.
    """
    if not isinstance(prop, dict):
        return False
    for branch in (*prop.get("anyOf", ()), *prop.get("oneOf", ())):
        if _accepts_string_array(branch):
            return True
    if prop.get("type") != "array":
        return False
    items = prop.get("items")
    # ``list[str]`` renders as items.type == "string"; an untyped list is
    # also accepted since every split product is a str.
    return not isinstance(items, dict) or items.get("type") in (None, "string")


def split_scalar_list(value: str) -> list[str]:
    """Split a scalar string into the list an array parameter expects.

    Handles the three shapes agents actually emit, in precedence order:

    * a JSON array literal — ``'["a", "b"]'``
    * a delimited string — ``"a,b"`` / ``"a, b"``
    * a whitespace-delimited string — ``"a b"``

    Tags in practice are slugs (``kv-cache``, ``model-loading``), so
    whitespace is a delimiter rather than part of one tag. A string with
    neither separator becomes a single-element list. Empty and
    whitespace-only input yields ``[]``, which every array parameter here
    treats as "unset".
    """
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError:
            pass  # Malformed literal: fall through to delimiter splitting.
        else:
            if isinstance(parsed, list):
                return [s for s in (str(item).strip() for item in parsed) if s]
    parts = text.split(",") if "," in text else text.split()
    return [s for s in (part.strip() for part in parts) if s]


def normalize_arguments(
    arguments: dict[str, Any],
    *,
    schema: dict[str, Any],
    tool_name: str = "tool",
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(normalized_arguments, notes)`` for one tool call.

    Pure and synchronous so the rules are testable without a server.
    ``notes`` records each rewrite for debug logging; an empty list means
    the call was already canonical, which is the overwhelmingly common
    case and costs one dict scan.

    Raises :class:`ToolError` when a call supplies both an alias and its
    canonical parameter with different values, or two aliases for the
    same target. Guessing which one the caller meant would silently
    discard an argument, so the ambiguity is surfaced instead.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not arguments:
        return arguments, []

    notes: list[str] = []
    renamed: dict[str, Any] = {}
    claimed: dict[str, str] = {}  # canonical -> alias that claimed it

    for key, value in arguments.items():
        if key in properties:
            renamed[key] = value
            continue
        target = next((c for c in _candidates(key) if c in properties), None)
        if target is None:
            # Not a recognised synonym. Leave it untouched so pydantic's
            # own "Unexpected keyword argument" still teaches the caller
            # the real parameter name.
            renamed[key] = value
            continue
        if target in arguments:
            if arguments[target] == value:
                notes.append(f"dropped redundant {key!r} (duplicate of {target!r})")
                continue
            raise ToolError(
                f"{tool_name}: got both {key!r} and {target!r} with different values "
                f"({value!r} vs {arguments[target]!r}). {key!r} is an alias for "
                f"{target!r} — pass only {target!r}."
            )
        if target in claimed:
            raise ToolError(
                f"{tool_name}: {key!r} and {claimed[target]!r} are both aliases for "
                f"{target!r}. Pass only {target!r}."
            )
        claimed[target] = key
        renamed[target] = value
        notes.append(f"renamed {key!r} -> {target!r}")

    # Type coercion runs after renaming so a call like ``tag="a,b"`` is
    # renamed to ``tags`` and then split in the same pass.
    out: dict[str, Any] = {}
    for key, value in renamed.items():
        if isinstance(value, str) and _accepts_string_array(properties.get(key)):
            coerced = split_scalar_list(value)
            out[key] = coerced
            notes.append(f"split {key!r} string into {len(coerced)} item(s)")
            continue
        out[key] = value
    return out, notes


class ArgumentNormalizationMiddleware(Middleware):
    """Rewrite near-miss tool arguments into their declared shape.

    Sits ahead of pydantic validation so the published schema can stay
    narrow. Schemas are resolved lazily on the first call to each tool
    and cached by name, keeping steady-state cost to one dict lookup.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}

    async def _schema_for(self, context: MiddlewareContext[Any], name: str) -> dict[str, Any]:
        """Input schema for ``name``, or ``{}`` when it cannot be resolved.

        An unresolvable tool (unknown name, hashed-name dispatch, a
        provider that raises) yields ``{}``, which makes
        :func:`normalize_arguments` a pass-through — normalization must
        never be the reason a call fails to dispatch.
        """
        cached = self._schemas.get(name)
        if cached is not None:
            return cached
        schema: dict[str, Any] = {}
        server = getattr(context.fastmcp_context, "fastmcp", None)
        if server is not None:
            try:
                tool = await server.get_tool(name)
            except Exception:
                tool = None
            parameters = getattr(tool, "parameters", None)
            if isinstance(parameters, dict):
                schema = parameters
        self._schemas[name] = schema
        return schema

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        arguments = context.message.arguments
        if not arguments:
            return await call_next(context)
        name = context.message.name
        schema = await self._schema_for(context, name)
        if not schema:
            return await call_next(context)
        normalized, notes = normalize_arguments(arguments, schema=schema, tool_name=name)
        if not notes:
            return await call_next(context)
        _LOG.debug("normalized %s arguments: %s", name, "; ".join(notes))
        message = context.message.model_copy(update={"arguments": normalized})
        return await call_next(context.copy(message=message))


__all__ = [
    "ArgumentNormalizationMiddleware",
    "normalize_arguments",
    "split_scalar_list",
]
