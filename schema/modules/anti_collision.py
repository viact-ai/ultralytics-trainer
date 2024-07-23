from pydantic import BaseModel


class AntiCollisionParam(BaseModel):
    classes: list | None = None
    PERSON_INDEX: int = 5
    WHEEL_INDEX: int = 7
    DISTANCE_THRESHOLD: int = 300


ANTI_COLLISION_ALLOW_CHANGES = [
    "DISTANCE_THRESHOLD",
]
