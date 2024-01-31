from enum import Enum
from typing import Dict, Union


def get_value(defined_type: Enum,
              value: Union[str, int]):
    for defined in defined_type:
        flag = False
        if isinstance(defined.value, int) \
                and isinstance(value, int):
            if defined.value == value:
                flag = True
        elif isinstance(defined.value, str) \
                and isinstance(value, str):
            if defined.value.lower() == value.lower():
                flag = True
        if flag:
            return defined

    return None


class BaseType(str, Enum):

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self):
        return str(self.value)


class BaseClasses(BaseType):
    CANVAS = "canvas"
    FENCE = "fence"
    HOLE = "hole"
    HOOK = "hook"
    PERSON = "person"


class ModuleType(BaseType):
    DANGER_ZONE = "danger-zone"
    NO_HELMET_DETECTION = "no-helmet-detection"
    LIFTING_LOAD_DANGER_ZONE = "lifting-load-danger-zone"
    SAFE_LIFTING = "safe-lifting"
    OPEN_EDGE = "open-edge"
    PERSON_NEAR_FENCE = "person-near-fence"
    VEHICLE_COUNTING = "vehicle-counting"
    TRAFFIC_JAM = "traffic-jam"
    MOTION_DETECTION = "motion-detection"


class ModelingType(BaseType):
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    CLASSIFICATION = "classification"


MAPPING_MODULE_TO_MODELING: Dict[str, str] = {
    ModuleType.DANGER_ZONE: ModelingType.OBJECT_DETECTION,
    ModuleType.NO_HELMET_DETECTION: ModelingType.OBJECT_DETECTION,
    ModuleType.LIFTING_LOAD_DANGER_ZONE: ModelingType.OBJECT_DETECTION,
    ModuleType.SAFE_LIFTING: ModelingType.OBJECT_DETECTION,
    ModuleType.OPEN_EDGE: ModelingType.INSTANCE_SEGMENTATION,
    ModuleType.PERSON_NEAR_FENCE: ModelingType.INSTANCE_SEGMENTATION,
    ModuleType.VEHICLE_COUNTING: ModelingType.OBJECT_DETECTION,
    ModuleType.TRAFFIC_JAM: ModelingType.OBJECT_DETECTION,
    ModuleType.MOTION_DETECTION: ModelingType.OBJECT_DETECTION,
}


DEFAULT_ALERT_STRING: Dict[str, Union[str, dict]] = {
    ModuleType.DANGER_ZONE: "ALERT: Detect object inside danger-zone",
    ModuleType.NO_HELMET_DETECTION: "ALERT: No helmet detection",
    ModuleType.LIFTING_LOAD_DANGER_ZONE: "ALERT: Detect object inside zone projection",
    ModuleType.SAFE_LIFTING: {
        BaseClasses.HOOK: "ALERT: Not safe due to the object not stopping to stabilize",
        BaseClasses.PERSON: "ALERT: Unsafe lifting due to people standing too close to the object"
    },
    ModuleType.OPEN_EDGE: "ALERT: No covered fence",
    ModuleType.PERSON_NEAR_FENCE:  "ALERT: Person near fence",
    ModuleType.TRAFFIC_JAM: "ALERT: Traffic jam is deteced in zone"
}


MODULE_CLASSES = {
    ModuleType.DANGER_ZONE: [BaseClasses.PERSON
                             ],
    ModuleType.LIFTING_LOAD_DANGER_ZONE: [
        BaseClasses.PERSON,
        BaseClasses.HOOK,
    ],
    ModuleType.OPEN_EDGE: [
        BaseClasses.PERSON,
        BaseClasses.CANVAS,
        BaseClasses.HOLE,
        BaseClasses.FENCE,
    ],
    ModuleType.SAFE_LIFTING: [
        BaseClasses.HOOK,
        BaseClasses.PERSON,
    ],
    ModuleType.PERSON_NEAR_FENCE: [
        BaseClasses.PERSON,
        BaseClasses.FENCE,
        BaseClasses.HOLE,
        BaseClasses.CANVAS,
    ],
}
