from pydantic import BaseModel


class SafetyHelmetParam(BaseModel):
    PERSON_INDEX: int = 0
    NO_HELMET_INDEX: int = 6
    PERCENT_OVERLAP: float = 0.7


SAFETY_HELMET_ALLOW_CHANGES = [
    "PERSON_INDEX",
    "NO_HELMET_INDEX",
    "PERCENT_OVERLAP",
]


SAFETY_HELMET_ALERT_ALLOW_CHANGES = [
    "FPS",
    "DURATION",
    "PERCENTAGE_OF_ALERT_FRAMES",
    "SEND_ALERT_FREQUENT",
]
