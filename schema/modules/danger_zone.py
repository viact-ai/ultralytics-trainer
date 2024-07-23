from pydantic import BaseModel


class DangerZoneParam(BaseModel):
    alert_classes: list | None = None
