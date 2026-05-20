"""
Geometry models for spatial elements in documents.

Provides BoundingBox, Size, and Point models with support for multiple
coordinate systems (PDF points, image pixels, normalized 0-1).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CoordSystem = Literal["pdf", "image", "normalized"]


class Point(BaseModel):
    """A 2D point in a document coordinate space."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="X-coordinate")
    y: float = Field(..., description="Y-coordinate")


class Size(BaseModel):
    """Width and height of a page, image, or bounding box."""

    model_config = ConfigDict(frozen=True)

    width: float = Field(..., description="Width dimension")
    height: float = Field(..., description="Height dimension")


class BoundingBox(BaseModel):
    """Axis-aligned bounding box defined by left, top, right, bottom coordinates.

    The ``coord_system`` field disambiguates which coordinate space the values
    are expressed in:

    - ``"pdf"``        → PDF points (typically 1/72 inch)
    - ``"image"``      → image pixel coordinates
    - ``"normalized"`` → values in [0, 1] relative to page / image dimensions
    """

    model_config = ConfigDict(frozen=True)

    left: float = Field(..., description="Left edge coordinate")
    top: float = Field(..., description="Top edge coordinate")
    right: float = Field(..., description="Right edge coordinate")
    bottom: float = Field(..., description="Bottom edge coordinate")
    coord_system: CoordSystem = Field(
        default="pdf",
        description="Coordinate system used for the values",
    )
