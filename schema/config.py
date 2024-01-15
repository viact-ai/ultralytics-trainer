from datetime import datetime


class BaseModel:
    def to_dict(self):
        def object_to_dict(obj):
            if isinstance(obj, (int, str, float)):
                return obj
            if isinstance(obj, datetime):
                return str(obj)
            if isinstance(obj, list):
                return [object_to_dict(item) for item in obj]
            if isinstance(obj, dict):
                return {key: object_to_dict(value) for key, value in obj.items()}
            if hasattr(obj, '__dict__'):
                return object_to_dict(obj.__dict__)
            return obj

        return object_to_dict(self.__dict__)


class ExperimentConfig(BaseModel):
    def __init__(self,
                 exp: str = None,
                 **kwargs):
        self.exp = exp


class AlertConfig(BaseModel):
    def __init__(self,
                 alert_string: str = None,
                 **kwargs):
        self.alert_string = alert_string


class InferenceConfig(BaseModel):
    def __init__(self,
                 imgsz: list = None,
                 conf_thres: float = 0.1,
                 iou_thres: float = 0.1,
                 max_det: int = None,
                 device: str = "0",
                 classes: list = None,
                 agnostic_nms: bool = False,
                 line_thickness: int = 3,
                 hide_labels: bool = False,
                 hide_conf: bool = False,
                 half: bool = False,
                 use_tensorrt_model: bool = False,
                 use_trt_with_onnx: bool = False,
                 retina_masks: bool = None,
                 num_masks: int = None,
                 **kwargs):
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.use_tensorrt_model = use_tensorrt_model
        self.use_trt_with_onnx = use_trt_with_onnx
        self.retina_masks = retina_masks
        self.num_masks = num_masks


class ModelingConfig(BaseModel):
    def __init__(self,
                 arch: str = None,
                 type: str = None,
                 train: dict = None,
                 inference: InferenceConfig = None,
                 size: str = None,
                 **kwargs):
        self.arch = arch
        self.type = type
        self.train = train
        self.inference = InferenceConfig(**inference)
        self.size = size


class ModelingDefaultConfig(BaseModel):
    def __init__(self,
                 model: ModelingConfig = None,
                 alerts: AlertConfig = None,
                 exp: ExperimentConfig = None,
                 algorithm: dict = None):
        self.model = ModelingConfig(**model)
        self.alerts = AlertConfig(**alerts)
        self.exp = ExperimentConfig(**exp)
        self.algorithm = algorithm
        self.alow_change = {
            "inference": [
                "conf_thres",
                "iou_thres",
                "max_det"
            ],
            "algorithm": []
        }
