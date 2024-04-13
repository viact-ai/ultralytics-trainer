from typing import Union, List
from pydantic import BaseModel


class OutsideWalkingParam(BaseModel):
    filter_classes: Union[list, dict, None] = None
    CONFIDENCE_THRESHOLD: float = 0.5
    PERCENT_OVERLAP: float = 0.9


OUTSIDE_WALKING_ALLOW_CHANGES = [
    "filter_classes",
    "CONFIDENCE_THRESHOLD",
    "PERCENT_OVERLAP"
]
