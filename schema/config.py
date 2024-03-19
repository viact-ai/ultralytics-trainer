from enum import Enum
from typing import Optional, Union, List, Dict

from pydantic import BaseModel

from enums.config import ModelingType


class AlertConfig(BaseModel):
    alert_string: str


class InferenceConfig(BaseModel):
    imgsz: list = None
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 3000
    classes: list | None = None
    agnostic_nms: Optional[bool] = False
    half: Optional[bool] = False
    use_tensorrt_model: Optional[bool] = False
    use_trt_with_onnx: Optional[bool] = False
    retina_masks: Optional[bool] = False
    num_masks: Optional[int] = 32


class ModelConfig(BaseModel):
    id: str = None
    arch: Union[str, Enum] = None
    type: ModelingType = None
    inference: InferenceConfig = None
    size: Optional[str] = None
    weight_path: str = None
    label_path: str = None


class AllowChange(BaseModel):
    inference: Dict[str, list] = {"0": ["conf_threshold", "iou_threshold"]}
    algorithm: list = []


class ModelingConfig(BaseModel):
    model: Union[List[ModelConfig], ModelConfig] = None
    alerts: dict = None
    algorithm: dict | None = {}
    allow_change: AllowChange | None = {}
