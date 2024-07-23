from pydantic import BaseModel


class PPEDetectionParam(BaseModel):
    filter_classes: list | None = [3, 4, 5, 6, 7]
    alert_classes: list | None = [5, 6]
    classnames: dict | None = {
        "no_safety_helmet": 5,
        "no_safety_vest": 6,
    }
    NUM_ALERT_THRESHOLD: int = 3


PPE_DETECTION_ALLOW_CHANGES = [
    "NUM_ALERT_THRESHOLD",
]
