from pydantic import BaseModel


class SafeLiftingParam(BaseModel):
    project_classes: list
    alert_classes: list
    DEVIATION: float = -120
    RATIO: float = 2
    # safe distance threshold between person and the projected center
    DISTANCE_THRESHOLD: float = 3.0
    # confusion frames threshold
    MAX_CONFUSION_TIMES: float = 3
    # the height threshold which start/end processing
    LOW_HEIGHT_OBJECT_THRESH: float = 4.9
    HIGH_HEIGHT_OBJECT_THRESH: float = 5.1
    # the number of stop frame for stable status
    STABLE_FRAME_THRESH: float = 30

    PERSON_HEIGHT: float = 1.7
    PERSON_HEIGHT_PIXEL: int = 80
    LOW_PERSON_HEIGHT_PIXEL: int = 40
    RATIO_OBJECTION: float = 0.6
    MAX_Y_PROJECT: int = 480
    NUM_GRID_POINTS: int = 20


SAFE_LIFTING_ALLOW_CHANGES = [
    "DEVIATION",
    "RATIO",
    "DISTANCE_THRESHOLD",
    "MAX_CONFUSION_TIMES",
    "LOW_HEIGHT_OBJECT_THRESH",
    "HIGH_HEIGHT_OBJECT_THRESH",
    "PERSON_HEIGHT",
    "PERSON_HEIGHT_PIXEL",
    "LOW_PERSON_HEIGHT_PIXEL",
    "RATIO_OBJECTION",
    "MAX_Y_PROJECT",
    "NUM_GRID_POINTS"
]
