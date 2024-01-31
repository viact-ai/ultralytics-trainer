from pydantic import BaseModel
from .open_edge import Classes


class PersonNearFenceParam(BaseModel):
    classes: Classes | None = None

    MAX_PERSON_NEAREST_COUNT: float = 25
    PERCENT_PERSON_HEIGHT: float = 0.7
    BUFFER_EXPAND_SIZE: float = 60
    MIN_POLYGON_SQUARE: float = 10


PERSON_NEAR_FENCE_ALLOW_CHANGES = [
    "MAX_PERSON_NEAREST_COUNT",
    "PERCENT_PERSON_HEIGHT",
    "BUFFER_EXPAND_SIZE",
    "MIN_POLYGON_SQUARE",
]
