"""Stage-one synthetic-artifact provenance and evaluation for Chest_XR."""

CLASS_LABELS = {"covid": 0, "normal": 1, "pneumonia": 2}
MARKER_TYPES = ("square", "cross")
DEFAULT_MARKER_SIZE = 12
DEFAULT_MARKER_MARGIN = 4


def validate_marker_options(marker_type, size, margin):
    """Validate shared marker settings without importing imaging or ML packages."""
    if marker_type not in MARKER_TYPES:
        raise ValueError("marker_type must be one of {}".format(MARKER_TYPES))
    if isinstance(size, bool) or not isinstance(size, int) or size < 3:
        raise ValueError("marker size must be an integer >= 3 pixels")
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("marker margin must be a non-negative integer")
