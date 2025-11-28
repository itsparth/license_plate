from __future__ import annotations

from typing import List, Literal, Optional

from .base import BoundingBox, Box, Constraints, RenderContext, Widget
from .units import IntField


MainAxisAlignment = Literal["start", "center", "end", "space_between"]
CrossAxisAlignment = Literal["start", "center", "end"]
AxisSize = Literal["min", "max"]


class Padding(Widget):
    left: IntField = 0
    right: IntField = 0
    top: IntField = 0
    bottom: IntField = 0
    child: Widget

    def layout(self, constraints: Constraints, scale: float) -> Box:
        left = int(self.left * scale)
        right = int(self.right * scale)
        top = int(self.top * scale)
        bottom = int(self.bottom * scale)

        inner_constraints = Constraints(
            min_width=max(0, constraints.min_width - left - right),
            max_width=max(0, constraints.max_width - left - right),
            min_height=max(0, constraints.min_height - top - bottom),
            max_height=max(0, constraints.max_height - top - bottom),
        )
        inner_box = self.child.layout(inner_constraints, scale)
        width = inner_box.width + left + right
        height = inner_box.height + top + bottom
        return Box(x=0, y=0, width=width, height=height)

    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        left = int(self.left * ctx.scale)
        top = int(self.top * ctx.scale)
        child_box = Box(
            x=box.x + left,
            y=box.y + top,
            width=box.width - left,
            height=box.height - top,
        )
        return self.child.render(child_box, ctx)


class Row(Widget):
    children: List[Widget]
    gap: IntField = 0
    main_axis_alignment: MainAxisAlignment = "start"
    cross_axis_alignment: CrossAxisAlignment = "center"
    cross_axis_size: AxisSize = "min"

    def layout(self, constraints: Constraints, scale: float) -> Box:
        gap_px = int(self.gap * scale)
        x = 0
        max_height = 0
        for index, child in enumerate(self.children):
            child_constraints = Constraints(
                min_width=0,
                max_width=max(0, constraints.max_width - x),
                min_height=constraints.min_height,
                max_height=constraints.max_height,
            )
            child_box = child.layout(child_constraints, scale)
            x += child_box.width
            if index < len(self.children) - 1:
                x += gap_px
            max_height = max(max_height, child_box.height)
        width = min(constraints.max_width, x)
        if self.cross_axis_size == "max":
            height = constraints.max_height
        else:
            height = min(constraints.max_height, max_height)
        return Box(x=0, y=0, width=width, height=height)

    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        gap_px = int(self.gap * ctx.scale)
        total_children_width = 0
        child_boxes: List[Box] = []
        max_child_height = 0

        for index, child in enumerate(self.children):
            child_constraints = Constraints(
                min_width=0,
                max_width=box.width,
                min_height=0,
                max_height=box.height,
            )
            measured = child.layout(child_constraints, ctx.scale)
            child_boxes.append(measured)
            total_children_width += measured.width
            if index < len(self.children) - 1:
                total_children_width += gap_px
            max_child_height = max(max_child_height, measured.height)

        free_space = max(0, box.width - total_children_width)
        if self.main_axis_alignment == "start":
            start_x = box.x
        elif self.main_axis_alignment == "center":
            start_x = box.x + free_space // 2
        elif self.main_axis_alignment == "end":
            start_x = box.x + free_space
        else:  # space_between
            start_x = box.x
            if len(self.children) > 1:
                gap_px = gap_px + free_space // (len(self.children) - 1)

        align_height = box.height if self.cross_axis_size == "max" else max_child_height

        cursor_x = start_x
        all_boxes: List[BoundingBox] = []
        for index, (child, measured) in enumerate(zip(self.children, child_boxes)):
            if self.cross_axis_alignment == "start":
                child_y = box.y
            elif self.cross_axis_alignment == "center":
                child_y = box.y + (align_height - measured.height) // 2
            else:  # end
                child_y = box.y + (align_height - measured.height)
            placed = Box(
                x=cursor_x, y=child_y, width=measured.width, height=measured.height
            )
            all_boxes.extend(child.render(placed, ctx))
            cursor_x += measured.width
            if index < len(self.children) - 1:
                cursor_x += gap_px
        return all_boxes


class Column(Widget):
    children: List[Widget]
    gap: IntField = 0
    main_axis_alignment: MainAxisAlignment = "start"
    cross_axis_alignment: CrossAxisAlignment = "center"
    cross_axis_size: AxisSize = "min"

    def layout(self, constraints: Constraints, scale: float) -> Box:
        gap_px = int(self.gap * scale)
        y = 0
        max_width = 0
        for index, child in enumerate(self.children):
            child_constraints = Constraints(
                min_width=constraints.min_width,
                max_width=constraints.max_width,
                min_height=0,
                max_height=max(0, constraints.max_height - y),
            )
            child_box = child.layout(child_constraints, scale)
            y += child_box.height
            if index < len(self.children) - 1:
                y += gap_px
            max_width = max(max_width, child_box.width)
        if self.cross_axis_size == "max":
            width = constraints.max_width
        else:
            width = min(constraints.max_width, max_width)
        height = min(constraints.max_height, y)
        return Box(x=0, y=0, width=width, height=height)

    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        gap_px = int(self.gap * ctx.scale)
        total_children_height = 0
        child_boxes: List[Box] = []
        max_child_width = 0

        for index, child in enumerate(self.children):
            child_constraints = Constraints(
                min_width=0,
                max_width=box.width,
                min_height=0,
                max_height=box.height,
            )
            measured = child.layout(child_constraints, ctx.scale)
            child_boxes.append(measured)
            total_children_height += measured.height
            if index < len(self.children) - 1:
                total_children_height += gap_px
            max_child_width = max(max_child_width, measured.width)

        free_space = max(0, box.height - total_children_height)
        if self.main_axis_alignment == "start":
            start_y = box.y
        elif self.main_axis_alignment == "center":
            start_y = box.y + free_space // 2
        elif self.main_axis_alignment == "end":
            start_y = box.y + free_space
        else:  # space_between
            start_y = box.y
            if len(self.children) > 1:
                gap_px = gap_px + free_space // (len(self.children) - 1)

        align_width = box.width if self.cross_axis_size == "max" else max_child_width

        cursor_y = start_y
        all_boxes: List[BoundingBox] = []
        for index, (child, measured) in enumerate(zip(self.children, child_boxes)):
            if self.cross_axis_alignment == "start":
                child_x = box.x
            elif self.cross_axis_alignment == "center":
                child_x = box.x + (align_width - measured.width) // 2
            else:  # end
                child_x = box.x + (align_width - measured.width)
            placed = Box(
                x=child_x, y=cursor_y, width=measured.width, height=measured.height
            )
            all_boxes.extend(child.render(placed, ctx))
            cursor_y += measured.height
            if index < len(self.children) - 1:
                cursor_y += gap_px
        return all_boxes


class Align(Widget):
    horizontal: Literal["start", "center", "end"] = "center"
    vertical: Literal["start", "center", "end"] = "center"
    child: Widget

    def layout(self, constraints: Constraints, scale: float) -> Box:
        child_box = self.child.layout(constraints, scale)
        return Box(x=0, y=0, width=child_box.width, height=child_box.height)

    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        child_constraints = Constraints(
            min_width=0,
            max_width=box.width,
            min_height=0,
            max_height=box.height,
        )
        measured = self.child.layout(child_constraints, ctx.scale)

        if self.horizontal == "start":
            child_x = box.x
        elif self.horizontal == "center":
            child_x = box.x + (box.width - measured.width) // 2
        else:  # end
            child_x = box.x + (box.width - measured.width)

        if self.vertical == "start":
            child_y = box.y
        elif self.vertical == "center":
            child_y = box.y + (box.height - measured.height) // 2
        else:  # end
            child_y = box.y + (box.height - measured.height)

        placed = Box(x=child_x, y=child_y, width=measured.width, height=measured.height)
        return self.child.render(placed, ctx)


class Container(Widget):
    width: Optional[IntField] = None
    height: Optional[IntField] = None
    color: str = "#00000000"  # Transparent by default
    child: Optional[Widget] = None

    def layout(self, constraints: Constraints, scale: float) -> Box:
        # If explicit dimensions, use them (scaled)
        if self.width is not None:
            resolved_w = int(self.width * scale)
        elif self.child is not None:
            # Measure child to get natural width
            child_measured = self.child.layout(constraints, scale)
            resolved_w = child_measured.width
        else:
            resolved_w = constraints.min_width

        if self.height is not None:
            resolved_h = int(self.height * scale)
        elif self.child is not None:
            # Measure child to get natural height
            child_measured = self.child.layout(constraints, scale)
            resolved_h = child_measured.height
        else:
            resolved_h = constraints.min_height

        resolved_w = max(constraints.min_width, min(constraints.max_width, resolved_w))
        resolved_h = max(
            constraints.min_height, min(constraints.max_height, resolved_h)
        )

        return Box(x=0, y=0, width=resolved_w, height=resolved_h)

    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        ctx.draw.rectangle(
            [box.x, box.y, box.x + box.width, box.y + box.height],
            fill=self.color,
        )
        if self.child is None:
            return []
        child_box = Box(x=box.x, y=box.y, width=box.width, height=box.height)
        return self.child.render(child_box, ctx)
