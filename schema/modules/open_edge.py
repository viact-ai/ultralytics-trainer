from pydantic import BaseModel


class Classes(BaseModel):
    person: int
    canvas: int
    fence: int
    hole: int


class OpenEdgeParam(BaseModel):
    classes: Classes | None = None
    MAX_OPEN_EDGE_COUNT: float = 25
    PERCENT_OVERLAP: float = 0.9
    BUFFER_EXPAND_SIZE: float = 60
    MIN_POLYGON_SQUARE: float = 10
    DISTANCE_THRESHOLD: float = 75
    FENCE_AREA_OVERLAP_THRESHOLD: float = 2500


OPEN_EDGE_ALLOW_CHANGES = [
    "MAX_OPEN_EDGE_COUNT",
    "PERCENT_OVERLAP",
    "BUFFER_EXPAND_SIZE",
    "MIN_POLYGON_SQUARE",
    "DISTANCE_THRESHOLD",
    "FENCE_AREA_OVERLAP_THRESHOLD"
]
