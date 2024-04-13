from pydantic import BaseModel
from typing import Union


class PPEAlertClasses(BaseModel):
    # no_shoes: int
    no_safety_helmet: int
    no_safety_vest: int
    # person:
    # no_gloves: int


class PPEDetectionParam(BaseModel):
    filter_classes: Union[list, None] = None
    alert_classes:  Union[list, None] = None
    classnames: Union[PPEAlertClasses, None] = None
    NUM_ALERT_THRESHOLD: int = 3


PPE_ALLOW_CHANGES = [
    "filter_classes",
    "alert_classes",
    "classnames",
    "NUM_ALERT_THRESHOLD"
]
