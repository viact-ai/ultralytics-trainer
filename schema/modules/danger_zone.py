from pydantic import BaseModel


class DangerZoneParam(BaseModel):
    alert_classes: list | None = None


class LiftingLoadDangerZoneParam(BaseModel):
    version: str | None = "v2"
    project_classes: list | None = None
    alert_classes: list | None = None
    MAX_ADJUSTED_POINTS_RETIAN: int | None = 5
    PROJECTIONS_WIDTH: int | None = 150
    PROJECTIONS_HEIGHT: int | None = 150
    DEVIDE_VALUE: int | None = 2

    WIDTH_RATIO_THRESHOLD: float = 4  # for distance filter out
    CLASSIFY_CONFIDENCE_THRESHOLD: float = 0.5
    BOX_POSTPROCESSING_IOU_THRESHOLD: float = 0.2

    HOOK_RATIO_WIDTH_HEIGHT: float = 0.7
    HOOK_AREA_THRESHOLD: float = 7000


LIFTING_LOAD_DANGER_ZONE_ALLOW_CHANGES = ["MAX_ADJUSTED_POINTS_RETIAN",
                                          "PROJECTIONS_WIDTH",
                                          "PROJECTIONS_HEIGHT",
                                          "DEVIDE_VALUE",
                                          "WIDTH_RATIO_THRESHOLD",
                                          "CLASSIFY_CONFIDENCE_THRESHOLD",
                                          "BOX_POSTPROCESSING_IOU_THRESHOLD",
                                          "HOOK_RATIO_WIDTH_HEIGHT",
                                          "HOOK_AREA_THRESHOLD",
                                          "version"]
