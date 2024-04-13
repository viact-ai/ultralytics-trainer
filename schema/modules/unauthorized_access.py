from pydantic import BaseModel
from typing import Union


class UnauthorizedAccessParam(BaseModel):
    filter_classes: Union[list, None] = None
    accepted_classes: Union[list, None] = None
    vest_classes: Union[list] = None


UNAUTHORIZED_ACCESS_ALLOW_CHANGES = [
    "filter_classes",
    "accepted_classes",
]
