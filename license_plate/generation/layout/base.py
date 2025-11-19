from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, ConfigDict
from PIL import Image, ImageDraw


class Constraints(BaseModel):
    min_width: int
    max_width: int
    min_height: int
    max_height: int


class Box(BaseModel):
    x: int
    y: int
    width: int
    height: int


class BoundingBox(BaseModel):
    label: str
    x: int
    y: int
    width: int
    height: int


class RenderContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    draw: ImageDraw.ImageDraw
    root_width: int
    root_height: int


class Widget(BaseModel, ABC):
    @abstractmethod
    def layout(
        self, constraints: Constraints, root_width: int, root_height: int
    ) -> Box:
        raise NotImplementedError

    @abstractmethod
    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        raise NotImplementedError
