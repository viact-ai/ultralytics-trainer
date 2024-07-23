from pydantic import BaseModel


class UnauthorizedAccessParam(BaseModel):
    filter_classes: list | None = [0,]
    accepted_classes: list | None = [0,]


UNAUTHORIZED_ACCESS_ALLOW_CHANGES = [
    "filter_classes",
    "accepted_classes",
]
