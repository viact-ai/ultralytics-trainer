from pydantic import BaseModel
from typing import Union


class NotConveringShoesClasses(BaseModel):
    person: int
    no_boots: int


class NotCoveringShoesParam(BaseModel):
    classes: Union[NotConveringShoesClasses, None] = None
    PERCENT_OVERLAP: float = 0.9


NOT_COVERING_SHOES_ALLOW_CHANGES = [
    "classes",
    "PERCENT_OVERLAP"
]
