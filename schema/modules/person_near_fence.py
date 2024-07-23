from pydantic import BaseModel

from .open_edge import Classes


class PersonNearFenceParam(BaseModel):
    classes: Classes | None = None
    PERCENT_OVERLAP: float = 0.9
    BUFFER_EXPAND_SIZE: float = 60.0
    MIN_POLYGON_SQUARE: float = 10.0
    DISTANCE_THRESHOLD: float = 25.0
    PERCENT_PERSON_HEIGHT: float = 0.6


PERSON_NEAR_FENCE_ALLOW_CHANGES = [
    "PERCENT_OVERLAP",
    "BUFFER_EXPAND_SIZE",
    "MIN_POLYGON_SQUARE",
    "DISTANCE_THRESHOLD",
    "PERCENT_PERSON_HEIGHT",
]


PERSON_NEAR_FENCE_ALERT_ALLOW_CHANGES = [
    "FPS",
    "DURATION",
    "PERCENTAGE_OF_ALERT_FRAMES",
    "SEND_ALERT_FREQUENT",
]
