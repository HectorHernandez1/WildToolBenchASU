"""AV — Argument Validator.

Pure-Python validator. Checks tool-call arguments against the tool's JSON
schema, flags missing required parameters and undefined extras, and applies
format normalizers (ISO date canonicalization, numeric coercion).

Targets AFF (Argument Fidelity) — 175 errors. We re-use the same
normalization rules already shipped in `wtb/checker_utils.py` so that
validator behavior matches what the scorer accepts.
"""
from __future__ import annotations

import re
from typing import Any


_DATE_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")


def normalize_date(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    m = _DATE_RE.match(value.strip())
    if not m:
        return value
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def coerce_numeric(value: Any, target_type: str | None) -> Any:
    if target_type not in {"integer", "number"}:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            if target_type == "integer":
                return int(value)
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


class ArgumentValidator:
    """Validates and normalizes tool-call arguments."""

    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_schemas: list[dict],
    ) -> tuple[bool, list[str], dict[str, Any]]:
        """Return (ok, errors, repaired_args)."""
        if not isinstance(arguments, dict):
            return False, ["arguments_not_dict"], {}

        # Find the matching schema
        schema = None
        for t in tool_schemas:
            fn = t.get("function", t)
            if fn.get("name") == tool_name:
                schema = fn.get("parameters", {}) or {}
                break

        if schema is None:
            # Tool not found → can't validate; pass through
            return True, [], dict(arguments)

        properties = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []

        repaired: dict[str, Any] = {}
        errors: list[str] = []

        # Copy + normalize known params, drop unknown
        for k, v in arguments.items():
            if k in properties:
                spec = properties[k]
                target_type = spec.get("type") if isinstance(spec, dict) else None
                # Format normalizers
                if isinstance(v, str):
                    v = normalize_date(v)
                if target_type:
                    v = coerce_numeric(v, target_type)
                repaired[k] = v
            else:
                errors.append(f"unknown_param:{k}")
                # Drop unknown params (lenient repair)

        # Check required
        missing = [r for r in required if r not in repaired]
        if missing:
            errors.append(f"missing_required:{','.join(missing)}")

        ok = not missing  # unknown params are normalized away (warning, not failure)
        return ok, errors, repaired
