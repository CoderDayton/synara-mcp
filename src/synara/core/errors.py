"""Domain error hierarchy. Tool layers translate these to MCP error responses."""

from __future__ import annotations


class SynaraError(Exception):
    """Base for every error raised by Synara service code."""


class NotFoundError(SynaraError):
    """Requested entity does not exist."""


class AlreadyExistsError(SynaraError):
    """Entity with the given identifier already exists."""


class ValidationError(SynaraError):
    """Caller supplied malformed or out-of-range input."""
