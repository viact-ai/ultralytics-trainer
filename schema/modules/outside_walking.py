from pydantic import BaseModel


class OutsideWalkingParam(BaseModel):
    filter_classes: list | None = [0,]
    accepted_classes: list | None = [0,]
    CONFIDENCE_THRESHOLD: float = 0.5
    PERCENT_OVERLAP: float = 0.9


OUTSIDE_WALKING_ALLOW_CHANGES = [
    "CONFIDENCE_THRESHOLD",
    "PERCENT_OVERLAP",
    "filter_classes",
    "accepted_classes",
]
