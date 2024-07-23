from enum import Enum
from typing import Any, Dict, Union


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
    CLASS_0 = "0"
    CLASS_1 = "1"
    BLACK = "black"
    BLUE = "blue"
    BOOM_LIFT = "boom_lift"
    BUCKET = "bucket"
    BULLDOZER = "bulldozer/backhoe/loader"
    BUS = "bus"
    CANVAS = "canvas"
    CAR = "car"
    CONCRETE_MIXER = "concrete_mixer"
    EXCAVATOR = "excavator"
    FACE = "face"
    FENCE = "fence"
    FRAME = "frame"
    GREEN = "green"
    HELMET = "helmet"
    HOLE = "hole"
    HOOK = "hook"
    HOUSEHOLD_SHELTER = "household shelter"
    MACHINERY = "machinery"
    MOBILE_CRANE = "mobile_crane"
    MOTORCYCLE = "motorcycle"
    NO_HELMET = "no_helmet"
    NO_SHOE = "no_shoe"
    NO_VEST = "no_vest"
    ORANGE = "orange"
    OTHERS = "others"
    PERSON = "person"
    PRECAST_PLANKS = "precast plank"
    RED = "red"
    ROLLER = "roller"
    SCISSOR_LIFT = "scissor_lift"
    SHOE = "shoe"
    STEEL_PLATE = "steel plate"
    STICK = "stick"
    TRUCK = "truck"
    UNDERFINE_IMG = "underfine_img"
    VAN = "van"
    VEST = "vest"
    WALL = "wall"
    WHEEL = "wheel"



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
    ANTI_COLLISION = "anti-collision"
    PPE_DETECTION = "ppe-detection"
    SAFETY_HELMET = "safety-helmet"
    SAFETY_VEST = "safety-vest"
    OUTSIDE_WALKING = "outside-walking"
    NO_COVERING_SHOES = "no-covering-shoes"
    UNAUTHORIZED_ACCESS = "unauthorized-access"
    ILLEGAL_PARKING = "illegal-parking"



class ModelingType(BaseType):
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    CLASSIFICATION = "classification"


class YOLOTasks(BaseType):
    DETECT = "detect"
    SEGMENT = "segment"
    CLASSIFY = "classify"


MAPPING_YOLO_TASK_TO_MODELING: Dict[Any, Any] = {
    YOLOTasks.DETECT: ModelingType.OBJECT_DETECTION,
    YOLOTasks.SEGMENT: ModelingType.INSTANCE_SEGMENTATION,
    YOLOTasks.CLASSIFY: ModelingType.CLASSIFICATION
}

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
    ModuleType.ANTI_COLLISION: ModelingType.OBJECT_DETECTION,
    ModuleType.PPE_DETECTION: ModelingType.OBJECT_DETECTION,
    ModuleType.SAFETY_HELMET: ModelingType.OBJECT_DETECTION,
    ModuleType.SAFETY_VEST: ModelingType.OBJECT_DETECTION,
    ModuleType.OUTSIDE_WALKING: ModelingType.OBJECT_DETECTION,
    ModuleType.NO_COVERING_SHOES: ModelingType.OBJECT_DETECTION,
    ModuleType.UNAUTHORIZED_ACCESS: ModelingType.OBJECT_DETECTION,
    ModuleType.ILLEGAL_PARKING: ModelingType.OBJECT_DETECTION,
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
    ModuleType.TRAFFIC_JAM: "ALERT: Traffic jam is deteced in zone",
    ModuleType.ANTI_COLLISION: "ALERT: Collision is deteced in zone",
    ModuleType.PPE_DETECTION: "ALERT: No PPE detected",
    ModuleType.SAFETY_HELMET: "ALERT: No helmet detected",
    ModuleType.SAFETY_VEST: "ALERT: No vest detected",
    ModuleType.OUTSIDE_WALKING: "ALERT: Walking outside the zone",
    ModuleType.NO_COVERING_SHOES: "ALERT: No covering shoes detected",
    ModuleType.UNAUTHORIZED_ACCESS: "ALERT: Unauthorized person access detected",
    ModuleType.ILLEGAL_PARKING: "ALERT: Illegal parking detected",
}


MODULE_CLASSES = {
    ModuleType.DANGER_ZONE: [BaseClasses.PERSON
                             ],
    ModuleType.LIFTING_LOAD_DANGER_ZONE: {
        "detect": [
            BaseClasses.PERSON,
            BaseClasses.HOOK,
            BaseClasses.FRAME,
            BaseClasses.WALL,
            BaseClasses.BUCKET,
            BaseClasses.PRECAST_PLANKS,
            BaseClasses.STICK,
            BaseClasses.HOUSEHOLD_SHELTER,
            BaseClasses.STEEL_PLATE,
        ],
        "classify": [
            BaseClasses.CLASS_0,
            BaseClasses.CLASS_1,
        ],
    },
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
    ModuleType.ANTI_COLLISION: [
        BaseClasses.CAR,
        BaseClasses.VAN,
        BaseClasses.BUS,
        BaseClasses.TRUCK,
        BaseClasses.MOTORCYCLE,
        BaseClasses.PERSON,
        BaseClasses.FACE,
        BaseClasses.WHEEL,
        BaseClasses.EXCAVATOR,
        BaseClasses.CONCRETE_MIXER,
        BaseClasses.BULLDOZER,
        BaseClasses.ROLLER,
        BaseClasses.BOOM_LIFT,
        BaseClasses.SCISSOR_LIFT,
        BaseClasses.MOBILE_CRANE,
        BaseClasses.MACHINERY,
    ],
    ModuleType.PPE_DETECTION: [
        BaseClasses.PERSON,
        BaseClasses.SHOE,
        BaseClasses.NO_SHOE,
        BaseClasses.VEST,
        BaseClasses.NO_VEST,
        BaseClasses.HELMET,
        BaseClasses.NO_HELMET,
    ],
    ModuleType.SAFETY_HELMET: [
        BaseClasses.PERSON,
        BaseClasses.SHOE,
        BaseClasses.NO_SHOE,
        BaseClasses.VEST,
        BaseClasses.NO_VEST,
        BaseClasses.HELMET,
        BaseClasses.NO_HELMET,
    ],
    ModuleType.SAFETY_VEST: [
        BaseClasses.PERSON,
        BaseClasses.SHOE,
        BaseClasses.NO_SHOE,
        BaseClasses.VEST,
        BaseClasses.NO_VEST,
        BaseClasses.HELMET,
        BaseClasses.NO_HELMET,
    ],
    ModuleType.UNAUTHORIZED_ACCESS: {
        "detect": [BaseClasses.PERSON],
        "classify": [
            BaseClasses.BLACK,
            BaseClasses.BLUE,
            BaseClasses.GREEN,
            BaseClasses.ORANGE,
            BaseClasses.RED,
            BaseClasses.UNDERFINE_IMG,
        ],
    },
    ModuleType.ILLEGAL_PARKING: [
        BaseClasses.TRUCK,
    ],
    ModuleType.OUTSIDE_WALKING: {
        "detect": [
            BaseClasses.PERSON,
        ],
        "classify": [
            BaseClasses.ORANGE,
            BaseClasses.OTHERS,
        ]
    },
    ModuleType.NO_COVERING_SHOES: [
        BaseClasses.PERSON,
        BaseClasses.SHOE,
        BaseClasses.NO_SHOE,
    ]
}
