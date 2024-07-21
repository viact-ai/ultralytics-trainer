from pydantic import BaseModel


class LiftingLoadDangerZoneParam(BaseModel):
    WIDTH_RATIO_THRESHOLD: float = 8  # for distance filter out
    BOX_POSTPROCESSING_IOU_THRESHOLD: float = 0.2
    CLASS_NAMES: list[str] = [
        "person", "hook", "frame", "wall", \
        "bucket", "precast planks", "stick", \
        "household shelter", "steel plate"]
    CLASSIFY_CONFIDENCE_THRESHOLD: float = 0.
    PERSON_INDEX: int = 0
    HOOK_INDEX: int = 1
    ALERT_STRING: str = "WORKER IN LIFTING DANGER ZONE"


LIFTING_LOAD_DANGER_ZONE_ALLOW_CHANGES = [
    "WIDTH_RATIO_THRESHOLD",
    "BOX_POSTPROCESSING_IOU_THRESHOLD",
    "ALERT_STRING",
]
