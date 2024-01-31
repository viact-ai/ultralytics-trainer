from pydantic import BaseModel


class DangerZoneParam(BaseModel):
    alert_classes: list | None = None


class LiftingLoadDangerZoneParam(BaseModel):
    project_classes: list | None = None
    alert_classes: list | None = None
    MAX_ADJUSTED_POINTS_RETIAN: int | None = 5
    PROJECTIONS_WIDTH: int | None = 150
    PROJECTIONS_HEIGHT: int | None = 150
    DEVIDE_VALUE: int | None = 2


LIFTING_LOAD_DANGER_ZONE_ALLOW_CHANGES = ["MAX_ADJUSTED_POINTS_RETIAN",
                                          "PROJECTIONS_WIDTH",
                                          "PROJECTIONS_HEIGHT",
                                          "DEVIDE_VALUE"]
