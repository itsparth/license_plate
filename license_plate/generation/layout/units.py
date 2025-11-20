from __future__ import annotations

from typing import Literal, Union
import random

from pydantic import BaseModel
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated


UnitKind = Literal["px", "vw", "vh", "pct"]


class Unit(BaseModel):
    kind: UnitKind
    value: float

    def resolve(self, *, parent: float, root_width: float, root_height: float) -> float:
        if self.kind == "px":
            return self.value
        if self.kind == "vw":
            return root_width * self.value
        if self.kind == "vh":
            return root_height * self.value
        if self.kind == "pct":
            return parent * self.value
        raise ValueError(f"Unknown unit kind: {self.kind}")


def px(value: float) -> Unit:
    return Unit(kind="px", value=value)


def vw(value: float) -> Unit:
    return Unit(kind="vw", value=value)


def vh(value: float) -> Unit:
    return Unit(kind="vh", value=value)


def pct(value: float) -> Unit:
    return Unit(kind="pct", value=value)


UnitLike = Union[Unit, float, int, str]


def parse_unit(value: UnitLike) -> Unit:
    if isinstance(value, Unit):
        return value
    if isinstance(value, (int, float)):
        return px(float(value))

    if not isinstance(value, str):
        raise TypeError(f"Unsupported unit value type: {type(value)!r}")

    spec = value.strip().lower()
    if spec.endswith("px"):
        return px(float(spec[:-2]))
    if spec.endswith("vw"):
        return vw(float(spec[:-2]) / 100.0)
    if spec.endswith("vh"):
        return vh(float(spec[:-2]) / 100.0)
    if spec.endswith("%"):
        return pct(float(spec[:-1]) / 100.0)

    # Bare number as string -> pixels
    return px(float(spec))


def u(value: UnitLike) -> Unit:
    return parse_unit(value)


UnitField = Annotated[Unit, BeforeValidator(parse_unit)]


def r(start: UnitLike, end: UnitLike) -> Unit:
    s = parse_unit(start)
    e = parse_unit(end)
    if s.kind != e.kind:
        raise ValueError(f"Range units must share kind (got {s.kind!r} vs {e.kind!r})")
    low = min(s.value, e.value)
    high = max(s.value, e.value)
    return Unit(kind=s.kind, value=random.uniform(low, high))
