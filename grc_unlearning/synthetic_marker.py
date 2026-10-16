"""Deterministic, in-memory PIL markers; input files and images stay unchanged."""

from PIL import Image, ImageDraw

from . import DEFAULT_MARKER_MARGIN, DEFAULT_MARKER_SIZE, validate_marker_options


def apply_marker(
    image,
    marker_type="square",
    size=DEFAULT_MARKER_SIZE,
    margin=DEFAULT_MARKER_MARGIN,
):
    """Return a copy with a black/white marker in the upper-right corner.

    Size and margin are measured in source-image pixels, before any transforms.
    A square has a white border and black center; a cross is black on a white
    square. No random state is used. Markers that do not fit raise ValueError.
    RGB, RGBA, and grayscale L images retain their mode; other modes become RGB.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    validate_marker_options(marker_type, size, margin)
    if size + margin > min(image.size):
        raise ValueError("marker size + margin must fit inside the image")

    result = image.copy() if image.mode in ("RGB", "RGBA", "L") else image.convert("RGB")
    white = 255 if result.mode == "L" else (255, 255, 255)
    black = 0 if result.mode == "L" else (0, 0, 0)
    if result.mode == "RGBA":
        white, black = (255, 255, 255, 255), (0, 0, 0, 255)

    left, top = result.width - margin - size, margin
    right, bottom = left + size - 1, top + size - 1
    draw = ImageDraw.Draw(result)
    draw.rectangle((left, top, right, bottom), fill=white)
    if marker_type == "square":
        border = max(1, size // 6)
        draw.rectangle((left + border, top + border, right - border, bottom - border), fill=black)
    else:
        thickness = max(1, size // 3)
        offset = (size - thickness) // 2
        draw.rectangle((left + offset, top, left + offset + thickness - 1, bottom), fill=black)
        draw.rectangle((left, top + offset, right, top + offset + thickness - 1), fill=black)
    return result
