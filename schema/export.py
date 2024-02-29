from typing import List, Optional, Union
from pydantic import BaseModel


class MODEL_TYPE(str):
    DETECTION = "DETECTION"
    SEGMENTATION = "SEGMENTATION"
    CLASSIFICATION = "CLASSIFICATION"


class ModelInfo(BaseModel):
    model_id: str
    model_arch: str
    model_type: str
    label_list: Optional[List[str]]
    imgsz: Optional[Union[int, list]] = 640
