"""Simple integer-based measurements with scale factor support."""

from __future__ import annotations

from typing import Union
import random

from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated


def parse_int(value: Union[int, float, str]) -> int:
    """Parse a value to integer."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(float(value.strip()))
    raise TypeError(f"Cannot parse {type(value)} to int")


IntField = Annotated[int, BeforeValidator(parse_int)]


def rand_int(low: int, high: int) -> int:
    """Random integer in range [low, high]."""
    return random.randint(min(low, high), max(low, high))
