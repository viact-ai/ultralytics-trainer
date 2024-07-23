from pydantic import BaseModel


class NoCoveringShoesParam(BaseModel):
    classes: list | None = None
    PERSON_CONFIDENCE_THRESHOLD: float = 0.5
    SHOE_CONFIDENCE_THRESHOLD: float = 0.25
    NO_SHOE_CONFIDENCE_THRESHOLD: float = 0.25
    PERCENT_OVERLAP: float = 0.9
    PERSON_INDEX: float = 0
    SHOE_INDEX: float = 1
    NO_SHOE_INDEX: float = 2


NO_COVERING_SHOES_ALLOW_CHANGES = [
    "PERSON_CONFIDENCE_THRESHOLD",
    "SHOE_CONFIDENCE_THRESHOLD",
    "NO_SHOE_CONFIDENCE_THRESHOLD",
    "PERCENT_OVERLAP",
]
