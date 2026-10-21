"""Drawing the film's frames: header, scene tiles, side panel, captions, cards and slates.

Every frame is laid out on a 1280 x 720 grid and scaled to the output size, so the master
and the deliverable show the same picture. The colours follow the campaign's figures: the
series blue for the state after a change, neutral ink for the state before, the
categorical yellow for placeholder footage and orange for a beat without footage. Fonts
are matplotlib's DejaVu files, installed with it, so a render looks the same everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

__all__ = ["Canvas", "PanelState", "TileSpec"]

RGB = tuple[int, int, int]
BG: RGB = (17, 19, 22)
BAND: RGB = (11, 12, 14)
PANEL_BG: RGB = (29, 32, 37)
TEXT: RGB = (238, 236, 230)
DIM: RGB = (150, 148, 142)
FAINT: RGB = (92, 92, 90)
BEFORE: RGB = (82, 81, 78)  # the campaign's neutral ink: the state before
AFTER: RGB = (42, 120, 214)  # the series blue: the state after
SERIES: RGB = (42, 120, 214)
PLACEHOLDER: RGB = (237, 161, 0)  # categorical yellow
MISSING: RGB = (235, 104, 52)  # categorical orange
GOOD: RGB = (27, 175, 122)  # categorical aqua
STORY: dict[str, RGB] = {"A": (42, 120, 214), "B": (27, 175, 122)}


def _font_dir() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


@lru_cache(maxsize=64)
def font(size: int, face: str = "sans") -> ImageFont.FreeTypeFont:
    name = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "mono": "DejaVuSansMono.ttf"}[
        face
    ]
    return ImageFont.truetype(str(_font_dir() / name), size)


def wrap(text: str, face: ImageFont.FreeTypeFont, width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}" if line else word
            if line and face.getlength(trial) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def ellipsize(text: str, face: ImageFont.FreeTypeFont, width: float) -> str:
    if face.getlength(text) <= width:
        return text
    while text and face.getlength(text + "…") > width:
        text = text[:-1]
    return text + "…"


@dataclass(frozen=True)
class PanelState:
    requirement: tuple[str, ...] = ()
    rule: str | None = None
    bar: str | None = None
    program: str | None = None
    steps: tuple[str, ...] = ()
    current: int | None = None  # the step in force; len(steps) when finished
    outcomes: tuple[str | None, ...] = ()  # per finished step: reached, timeout, ...
    # each count's text, detail line, and where it comes from (a table's public identifier
    # and SHA-256 prefix, or a literal count's stated source)
    counts: tuple[tuple[str, str | None, str | None], ...] = ()

    @property
    def empty(self) -> bool:
        return not (
            self.requirement or self.rule or self.bar or self.program or self.steps or self.counts
        )


@dataclass(frozen=True)
class TileSpec:
    label: str
    color: RGB


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


class Canvas:
    """The layout at one output size."""

    def __init__(self, width: int, height: int) -> None:
        if width * 9 != height * 16:
            raise ValueError("the film is 16:9")
        self.width, self.height = width, height
        self.k = height / 720.0

    def u(self, value: float) -> int:
        return round(value * self.k)

    def f(self, size: float, face: str = "sans") -> ImageFont.FreeTypeFont:
        return font(max(8, self.u(size)), face)

    # ---- regions, in 1280 x 720 units ----------------------------------------------------

    @property
    def scene(self) -> Box:
        """The scene area beside the side panel; `wide` is the whole width when there is none."""
        return Box(self.u(24), self.u(72), self.u(828), self.u(540))

    @property
    def wide(self) -> Box:
        return Box(self.u(24), self.u(72), self.u(1232), self.u(540))

    def area(self, wide: bool) -> Box:
        return self.wide if wide else self.scene

    @property
    def panel(self) -> Box:
        return Box(self.u(872), self.u(72), self.u(384), self.u(540))

    def tiles(self, n: int, wide: bool = False) -> list[Box]:
        """Image boxes for one scene view or a before-and-after pair (each below its header)."""
        area, header = self.area(wide), self.u(28)
        if n == 1:
            side = area.h - header
            return [Box(area.x + (area.w - side) // 2, area.y + header, side, side)]
        gap = self.u(12)
        side = min((area.w - gap) // 2, area.h - header)
        left = area.x + (area.w - 2 * side - gap) // 2
        top = area.y + (area.h - side - header) // 2 + header
        return [Box(left, top, side, side), Box(left + side + gap, top, side, side)]

    # ---- pieces --------------------------------------------------------------------------

    def blank(self) -> Image.Image:
        return Image.new("RGB", (self.width, self.height), BG)

    def header(
        self,
        image: Image.Image,
        story: str | None,
        story_name: str | None,
        title: str,
        subtitle: str | None,
        badges: Sequence[tuple[str, RGB]],
    ) -> None:
        draw = ImageDraw.Draw(image)
        x = self.u(24)
        color = STORY.get(story or "", DIM)
        if story:
            tag = f"STORY {story}" + (f"  ·  {story_name}" if story_name else "")
            draw.text((x, self.u(10)), tag.upper(), fill=color, font=self.f(13, "bold"))
        title_font = self.f(22, "bold")
        draw.text((x, self.u(30)), title, fill=TEXT, font=title_font)
        right = self.width - self.u(24)
        for text, fill in reversed(list(badges)):
            face = self.f(12, "bold")
            w = int(face.getlength(text)) + self.u(16)
            draw.rounded_rectangle(
                (right - w, self.u(14), right, self.u(38)), radius=self.u(4), fill=fill
            )
            draw.text((right - w + self.u(8), self.u(19)), text, fill=BG, font=face)
            right -= w + self.u(8)
        if subtitle:
            start = x + int(title_font.getlength(title)) + self.u(14)
            face = self.f(16)
            room = right - start - self.u(12)
            if room > self.u(80):
                draw.text((start, self.u(35)), ellipsize(subtitle, face, room), fill=DIM, font=face)

    def source_line(self, image: Image.Image, segments: Sequence[tuple[str, RGB]]) -> None:
        """One small line under the scene: what the footage is and where it came from."""
        draw = ImageDraw.Draw(image)
        face = self.f(12)
        x, right = self.u(24), self.u(24) + self.wide.w
        for text, color in segments:
            if x >= right:
                break
            text = ellipsize(text, face, right - x)
            draw.text((x, self.u(617)), text, fill=color, font=face)
            x += int(face.getlength(text)) + self.u(6)

    def caption(self, image: Image.Image, text: str | None) -> None:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, self.u(640), self.width, self.height), fill=BAND)
        if not text:
            return
        face = self.f(24)
        lines = wrap(text, face, self.width - self.u(160))[:2]
        line_h = self.u(31)
        top = self.u(640) + (self.u(80) - line_h * len(lines)) // 2
        for i, line in enumerate(lines):
            w = face.getlength(line)
            draw.text(((self.width - w) / 2, top + i * line_h), line, fill=TEXT, font=face)

    def tile_headers(
        self, image: Image.Image, boxes: Sequence[Box], tiles: Sequence[TileSpec]
    ) -> None:
        draw = ImageDraw.Draw(image)
        header = self.u(28)
        face = self.f(14, "bold")
        for box, tile in zip(boxes, tiles, strict=True):
            draw.rectangle((box.x, box.y - header, box.x + box.w - 1, box.y - 1), fill=tile.color)
            draw.text(
                (box.x + self.u(8), box.y - header + self.u(6)),
                ellipsize(tile.label, face, box.w - self.u(16)),
                fill=(255, 255, 255),
                font=face,
            )

    def side_panel(self, image: Image.Image, state: PanelState) -> None:
        if state.empty:
            return
        box = self.panel
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (box.x, box.y, box.x + box.w, box.y + box.h), radius=self.u(8), fill=PANEL_BG
        )
        pad = self.u(16)
        x, y = box.x + pad, box.y + pad
        width = box.w - 2 * pad
        label = self.f(12, "bold")
        body = self.f(15)
        small = self.f(13)
        line_h = self.u(21)

        def heading(text: str) -> None:
            nonlocal y
            draw.text((x, y), text, fill=DIM, font=label)
            y += self.u(20)

        def para(text: str, face: ImageFont.FreeTypeFont, fill: RGB, indent: int = 0) -> None:
            nonlocal y
            for line in wrap(text, face, width - indent):
                draw.text((x + indent, y), line, fill=fill, font=face)
                y += line_h

        if state.requirement or state.rule or state.bar:
            heading("REQUIREMENT")
            for goal in state.requirement:
                draw.text((x, y), "•", fill=TEXT, font=body)
                para(goal, body, TEXT, indent=self.u(14))
            if state.rule:
                para(f"Rule: {state.rule}", small, DIM)
            if state.bar:
                para(f"Bar: {state.bar}", small, DIM)
            y += self.u(14)
        if state.program or state.steps:
            heading("PROGRAM" + (f"  ·  {state.program.upper()}" if state.program else ""))
            for i, step in enumerate(state.steps):
                current = state.current == i
                done = state.current is not None and i < state.current
                outcome = state.outcomes[i] if i < len(state.outcomes) else None
                mark, mark_fill = f"{i + 1}", DIM
                if done:
                    good = outcome in (None, "reached")
                    # a tick says the step's stop condition fired, not that anything was
                    # delivered; so it stays neutral
                    mark, mark_fill = ("✓", DIM) if good else ("✗", MISSING)
                if current:
                    lines = wrap(step, body, width - self.u(30))
                    draw.rectangle(
                        (
                            x - self.u(8),
                            y - self.u(3),
                            x - self.u(5),
                            y + line_h * len(lines) - self.u(4),
                        ),
                        fill=SERIES,
                    )
                    mark, mark_fill = "▶", SERIES
                draw.text((x, y), mark, fill=mark_fill, font=body)
                fill = (
                    TEXT
                    if current
                    else (DIM if done else FAINT if state.current is not None else TEXT)
                )
                face = self.f(15, "bold") if current else body
                para(step, face, fill, indent=self.u(26))
                y += self.u(2)
            y += self.u(14)
        if state.counts:
            heading("COUNT" if len(state.counts) == 1 else "COUNTS")
            for text, detail, source in state.counts:
                para(text, self.f(24 if len(state.counts) == 1 else 20, "bold"), TEXT)
                y += self.u(2)
                if detail:
                    para(detail, small, DIM)
                if source:
                    para(source, self.f(11), FAINT)
                y += self.u(10)

    def card(
        self, title: str, subtitle: str | None, footnote: str | None, story: str | None
    ) -> Image.Image:
        image = self.blank()
        draw = ImageDraw.Draw(image)
        color = STORY.get(story or "", SERIES)
        face = self.f(44, "bold")
        lines = wrap(title, face, self.width - self.u(240))
        line_h = self.u(56)
        sub = self.f(22)
        sub_lines = wrap(subtitle, sub, self.width - self.u(300)) if subtitle else []
        height = line_h * len(lines) + (
            self.u(24) + self.u(32) * len(sub_lines) if sub_lines else 0
        )
        y = (self.u(600) - height) // 2
        draw.rectangle(
            (self.u(120), y - self.u(28), self.u(120) + self.u(64), y - self.u(22)), fill=color
        )
        for line in lines:
            draw.text((self.u(120), y), line, fill=TEXT, font=face)
            y += line_h
        y += self.u(24)
        for line in sub_lines:
            draw.text((self.u(120), y), line, fill=DIM, font=sub)
            y += self.u(32)
        if footnote:
            note = self.f(14)
            for i, line in enumerate(wrap(footnote, note, self.width - self.u(240))[:2]):
                draw.text((self.u(120), self.u(590) + i * self.u(20)), line, fill=FAINT, font=note)
        return image

    def slate(self, image: Image.Image, needs: str | None, title: str, wide: bool = False) -> None:
        draw = ImageDraw.Draw(image)
        box = self.area(wide)
        dash, gap = self.u(14), self.u(8)
        width = max(1, self.u(2))
        for x0 in range(box.x, box.x + box.w, dash + gap):
            x1 = min(x0 + dash, box.x + box.w)
            draw.line((x0, box.y, x1, box.y), fill=MISSING, width=width)
            draw.line((x0, box.y + box.h, x1, box.y + box.h), fill=MISSING, width=width)
        for y0 in range(box.y, box.y + box.h, dash + gap):
            y1 = min(y0 + dash, box.y + box.h)
            draw.line((box.x, y0, box.x, y1), fill=MISSING, width=width)
            draw.line((box.x + box.w, y0, box.x + box.w, y1), fill=MISSING, width=width)
        x, y = box.x + self.u(48), box.y + self.u(150)
        draw.text((x, y), "NO FOOTAGE YET", fill=MISSING, font=self.f(34, "bold"))
        y += self.u(58)
        draw.text((x, y), title, fill=TEXT, font=self.f(20, "bold"))
        y += self.u(36)
        if needs:
            face = self.f(18)
            for line in wrap(f"Needs: {needs}", face, box.w - self.u(96))[:7]:
                draw.text((x, y), line, fill=DIM, font=face)
                y += self.u(26)

    def still(
        self, image: Image.Image, picture: Image.Image, label: str | None, wide: bool = False
    ) -> None:
        box = self.area(wide)
        header = self.u(28) if label else 0
        room_w, room_h = box.w, box.h - header
        scale = min(room_w / picture.width, room_h / picture.height)
        w, h = max(1, int(picture.width * scale)), max(1, int(picture.height * scale))
        fitted = picture.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        x = box.x + (box.w - w) // 2
        y = box.y + header + (room_h - h) // 2
        image.paste(fitted, (x, y))
        if label:
            self.tile_headers(image, [Box(x, y, w, h)], [TileSpec(label, BEFORE)])

    def listing(self, image: Image.Image, title: str | None, text: str, wide: bool = False) -> None:
        box = self.area(wide)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (box.x, box.y, box.x + box.w, box.y + box.h), radius=self.u(8), fill=(24, 26, 30)
        )
        pad = self.u(20)
        y = box.y + pad
        if title:
            draw.text((box.x + pad, y), title, fill=TEXT, font=self.f(17, "bold"))
            y += self.u(34)
        room = box.y + box.h - pad - y
        width = box.w - 2 * pad
        source = text.expandtabs(4).splitlines() or [""]
        for size in (16, 15, 14, 13, 12, 11, 10):
            face = self.f(size, "mono")
            line_h = self.u(size * 1.35)
            lines = _wrap_mono(source, face, width)
            if line_h * len(lines) <= room or size == 10:
                break
        for line in lines[: max(1, room // line_h)]:
            draw.text((box.x + pad, y), line, fill=TEXT, font=face)
            y += line_h

    def messages(
        self,
        image: Image.Image,
        messages: Sequence[tuple[str, str]],
        shown: int,
        wide: bool = False,
    ) -> None:
        box = self.area(wide)
        draw = ImageDraw.Draw(image)
        y = box.y + self.u(10)
        name_face, body = self.f(14, "bold"), self.f(17)
        for i, (who, text) in enumerate(messages[:shown]):
            agent = "agent" in who.lower()
            color = SERIES if agent else MISSING
            indent = self.u(120) if agent else 0
            width = box.w - self.u(120)
            lines = wrap(text, body, width - self.u(32))
            h = self.u(24) * len(lines) + self.u(40)
            if y + h > box.y + box.h:
                break
            x = box.x + indent
            draw.rounded_rectangle((x, y, x + width, y + h), radius=self.u(10), fill=PANEL_BG)
            draw.rectangle((x, y + self.u(8), x + self.u(3), y + h - self.u(8)), fill=color)
            draw.text((x + self.u(16), y + self.u(8)), who, fill=color, font=name_face)
            for j, line in enumerate(lines):
                draw.text(
                    (x + self.u(16), y + self.u(30) + j * self.u(24)), line, fill=TEXT, font=body
                )
            y += h + self.u(12)
            del i


def _wrap_mono(lines: Sequence[str], face: ImageFont.FreeTypeFont, width: float) -> list[str]:
    """Lines of a listing wrapped at the width, continuations indented by two spaces."""
    import textwrap

    columns = max(10, int(width // max(1.0, face.getlength("M"))))
    out: list[str] = []
    for line in lines:
        if len(line) <= columns:
            out.append(line)
            continue
        indent = " " * (len(line) - len(line.lstrip()) + 2)
        out += textwrap.wrap(line, columns, subsequent_indent=indent, break_long_words=True) or [""]
    return out


def fit_tile(
    frame: npt.NDArray[np.uint8], crop: tuple[int, int, int, int], side: int
) -> npt.NDArray[np.uint8]:
    """The cropped view scaled into a square tile, letterboxed if it is not square."""
    x, y, w, h = crop
    part = Image.fromarray(np.ascontiguousarray(frame[y : y + h, x : x + w]))
    if w == h:
        if part.size != (side, side):
            part = part.resize((side, side), Image.Resampling.BICUBIC)
        return np.asarray(part, dtype=np.uint8)
    scale = side / max(w, h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    tile = Image.new("RGB", (side, side), BG)
    tile.paste(
        part.resize(size, Image.Resampling.BICUBIC), ((side - size[0]) // 2, (side - size[1]) // 2)
    )
    return np.asarray(tile, dtype=np.uint8)
