from enum import Enum
from typing import Dict


class BaseType(str, Enum):

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self):
        return str(self.value)
    
class ModuleType(BaseType):
    DANGER_ZONE = "danger-zone"
    SAFETY_HELMET = "safety-helmet"


class ModelingType(BaseType):
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"


MAPPING_MODULE_TO_MODELING: Dict[str, str] = {
    ModuleType.DANGER_ZONE: ModelingType.OBJECT_DETECTION,
    ModuleType.SAFETY_HELMET: ModelingType.OBJECT_DETECTION
}


DEFAULT_ALERT_STRING: Dict[str, str] = {
    ModuleType.DANGER_ZONE: "ALERT: Detect object inside Danger-zone",
    ModuleType.SAFETY_HELMET: "ALERT: No-safety-helmet detected"
}
