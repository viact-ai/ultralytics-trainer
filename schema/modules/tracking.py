from typing import Union
from pydantic import BaseModel


class MotionDetectionParam(BaseModel):
    classes: Union[list, dict]
    LOWER_MOTION_THRESHOLD: int = 15
    UPPER_LOWER_THRESHOLD: int = 100
    MAX_NUMBER_OF_CHECKING_POINTS: int = 30
    USE_ADAPTIVE_MOTION_THRESHOLD: bool = True
    USE_FOUR_CORNER_CHECK: bool = True


class TrafficJamParam(MotionDetectionParam):
    LINGER_TIME_THRESHOLD: int = 12
    PERCENT_THRESHOLD: float = 0.25


class SpeedEstimation(MotionDetectionParam):
    pass


class IllegalParkingParam(TrafficJamParam):
    pass


MOTION_DETECTION_ALLOW_CHANGES = [
    "LOWER_MOTION_THRESHOLD",
    "UPPER_LOWER_THRESHOLD",
    "MAX_NUMBER_OF_CHECKING_POINTS",
    "USE_ADAPTIVE_MOTION_THRESHOLD",
    "USE_FOUR_CORNER_CHECK"
]

TRAFFIC_JAM_ALLOW_CHANGES = MOTION_DETECTION_ALLOW_CHANGES + [
    "LINGER_TIME_THRESHOLD",
    "PERCENT_THRESHOLD"
]


ILLEGAL_PARKING_ALLOW_CHANGES = TRAFFIC_JAM_ALLOW_CHANGES
