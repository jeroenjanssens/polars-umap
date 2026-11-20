from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

PLUGIN_PATH = Path(__file__).parent


def parse_into_expr(value: pl.Expr | str | pl.Series) -> pl.Expr:
    """Turn a column name, Series or expression into an expression."""
    if isinstance(value, pl.Expr):
        return value
    if isinstance(value, str):
        return pl.col(value)
    if isinstance(value, pl.Series):
        return pl.lit(value)
    msg = f"expected a column name, expression or Series, got {type(value).__name__}"
    raise TypeError(msg)


def register(
    function_name: str,
    args: list[pl.Expr],
    kwargs: dict[str, Any],
) -> pl.Expr:
    """Register a call to one of the non-elementwise plugin functions."""
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name=function_name,
        args=args,
        kwargs=kwargs,
        is_elementwise=False,
        changes_length=False,
        returns_scalar=False,
        input_wildcard_expansion=True,
    )


def check_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer, got {value!r}"
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name} must be at least {minimum}, got {value}"
        raise ValueError(msg)
    return value


def check_optional_int(name: str, value: Any, minimum: int) -> int | None:
    return None if value is None else check_int(name, value, minimum)


def check_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a number, got {value!r}"
        raise TypeError(msg)
    return float(value)


def check_optional_float(name: str, value: Any) -> float | None:
    return None if value is None else check_float(name, value)


def check_choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    if value not in choices:
        msg = f"{name} must be one of {choices}, got {value!r}"
        raise ValueError(msg)
    return str(value)
