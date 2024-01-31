from ultralytics import YOLO
from pathlib import Path
import json

from schema.modules.danger_zone import (
    LIFTING_LOAD_DANGER_ZONE_ALLOW_CHANGES,
    DangerZoneParam,
    LiftingLoadDangerZoneParam)
from schema.modules.open_edge import OPEN_EDGE_ALLOW_CHANGES, OpenEdgeParam, Classes
from schema.modules.person_near_fence import PERSON_NEAR_FENCE_ALLOW_CHANGES, PersonNearFenceParam
from schema.modules.safe_lifting import SAFE_LIFTING_ALLOW_CHANGES, SafeLiftingParam
from schema.modules.tracking import (
    MOTION_DETECTION_ALLOW_CHANGES,
    TRAFFIC_JAM_ALLOW_CHANGES,
    MotionDetectionParam,
    TrafficJamParam)
from schema.config import (
    ModelingConfig,
    ModelConfig,
    InferenceConfig,
    AlertConfig,
    AllowChange)
from enums.config import (
    BaseClasses,
    ModuleType,
    MAPPING_MODULE_TO_MODELING,
    MODULE_CLASSES,
    DEFAULT_ALERT_STRING)


def label_list_to_txt(label_list: list[str],
                      labels_txt_path: Path,
                      model_class_names: dict = None) -> bool:
    names = list(model_class_names.values())
    intersection = set(names).intersection(label_list)
    flag = len(intersection) == len(
        label_list) and len(intersection) == len(names)
    if flag:
        labels_txt = "\n".join(names)
        with open(labels_txt_path, "w") as f:
            f.write(labels_txt)
    return names


def get_class_names(
    use_case: str, use_case_classes: dict, base_class_names: list[str]
) -> list[str]:
    if use_case not in use_case_classes:
        raise ValueError("use_case not supported")

    class_names = [
        (base_class_names.index(cls_name), cls_name)
        for cls_name in use_case_classes[use_case]
        if cls_name in base_class_names
    ]
    # sort ascending by index
    class_names = sorted(class_names, key=lambda x: x[0])
    return class_names


def export_to_onnx(model_path: str, format: str = 'onnx', imgsz: int = 640) -> str:
    '''
    Ultralytics ONNX args: imgsz, half, dynamic, simplify, opset
    Reference to the document for more details
    Example saved position:
        {workspace}
            /utils
                {model_weights}.onnx <- Ultralytics YOLO stored convert model here
                train_yolo.py
                ...
    '''
    model = YOLO(model=model_path)
    output_name = model.export(
        format=format,
        imgsz=imgsz,
        dynamic=True
    )
    names = model.names
    print(f"ONNX model stored at: {output_name}")
    return output_name, names


def get_algorithm_allow_change(ai_module: ModuleType):
    # if ai_module

    if ai_module == ModuleType.LIFTING_LOAD_DANGER_ZONE:
        return LIFTING_LOAD_DANGER_ZONE_ALLOW_CHANGES
    elif ai_module == ModuleType.OPEN_EDGE:
        return OPEN_EDGE_ALLOW_CHANGES
    elif ai_module == ModuleType.PERSON_NEAR_FENCE:
        return PERSON_NEAR_FENCE_ALLOW_CHANGES
    elif ai_module == ModuleType.SAFE_LIFTING:
        return SAFE_LIFTING_ALLOW_CHANGES
    elif ai_module == ModuleType.MOTION_DETECTION:
        return MOTION_DETECTION_ALLOW_CHANGES
    elif ai_module == ModuleType.TRAFFIC_JAM:
        return TRAFFIC_JAM_ALLOW_CHANGES
    else:
        return []


def check_class(ai_module: ModuleType,
                classes: list):
    count = 0
    neccessary_classes = MODULE_CLASSES[ai_module]
    for cls in neccessary_classes:
        if cls in classes:
            count += 1
    return count == len(neccessary_classes)


def get_configs(ai_module: ModuleType,
                classes: list):
    algo_config = {}
    alert_config = {}
    alert_str = "ALERT"
    if ai_module == ModuleType.DANGER_ZONE:
        algo_config = DangerZoneParam(
            alert_classes=[classes.index(str(BaseClasses.PERSON))])
        alert_str = DEFAULT_ALERT_STRING[ai_module]
    elif ai_module == ModuleType.LIFTING_LOAD_DANGER_ZONE:
        algo_config = LiftingLoadDangerZoneParam(
            project_classes=[classes.index(str(BaseClasses.HOOK))],
            alert_classes=[classes.index(str(BaseClasses.PERSON))]
        )
        alert_str = DEFAULT_ALERT_STRING[ai_module]

    elif ai_module == ModuleType.OPEN_EDGE:
        algo_config = OpenEdgeParam(
            classes=Classes(person=classes.index(str(BaseClasses.PERSON)),
                            canvas=classes.index(str(BaseClasses.CANVAS)),
                            fence=classes.index(str(BaseClasses.FENCE)),
                            hole=classes.index(str(BaseClasses.HOLE)))
        )
        alert_str = DEFAULT_ALERT_STRING[ai_module]
    elif ai_module == ModuleType.PERSON_NEAR_FENCE:
        algo_config = OpenEdgeParam(
            classes=Classes(person=classes.index(str(BaseClasses.PERSON)),
                            canvas=classes.index(str(BaseClasses.CANVAS)),
                            fence=classes.index(str(BaseClasses.FENCE)),
                            hole=classes.index(str(BaseClasses.HOLE)))
        )
        alert_str = DEFAULT_ALERT_STRING[ai_module]
    elif ai_module == ModuleType.SAFE_LIFTING:
        algo_config = SafeLiftingParam(
            project_classes=[classes.index(str(BaseClasses.HOOK))],
            alert_classes=[classes.index(str(BaseClasses.PERSON))]
        )
        alert_key = "alert_" + str(algo_config.project_classes[0])
        alert_config[alert_key] = DEFAULT_ALERT_STRING[ai_module][BaseClasses.HOOK]

        alert_key = "alert_" + str(algo_config.alert_classes[0])
        alert_config[alert_key] = DEFAULT_ALERT_STRING[ai_module][BaseClasses.PERSON]

    elif ai_module == ModuleType.MOTION_DETECTION:
        algo_config = MotionDetectionParam(
            classes=[i for i in range(len(classes))])

    elif ai_module == ModuleType.TRAFFIC_JAM:
        algo_config = TrafficJamParam(classes=[i for i in range(len(classes))])
        alert_str = DEFAULT_ALERT_STRING[ai_module]

    alert_config["alert_string"] = alert_str
    if not isinstance(algo_config, dict):
        algo_config = algo_config.model_dump()
    return algo_config, alert_config


def get_default_config(
    config_path: list[Path],
    imgsz: list = [640, 640],
    model_arch: str = "yolov5s",
    module: ModuleType = None,
    classes: list = None,
):
    # default_config = load_default_config(module=module)
    model_arch_version = None
    model_size = None
    if "yolov5" in model_arch:
        model_arch_version = "yolov5"
        model_size = model_arch.split("yolov5")[-1]
    elif "yolov8" in model_arch:
        model_arch_version = "yolov8"
        model_size = model_arch.split("yolov8")[-1]

    inference_config = InferenceConfig(
        imgsz=imgsz,
        classes=None
    )
    modeling_type = MAPPING_MODULE_TO_MODELING[module]

    # Update model config
    model_config = ModelConfig(
        arch=model_arch_version,
        type=modeling_type,
        inference=inference_config,
        size=model_size
    )
    allow_change = AllowChange(
        inference=["conf_threshold", "iou_threshold"],
        algorithm=get_algorithm_allow_change(ai_module=module)
    )
    algo_config, alert_config = get_configs(ai_module=module, classes=classes)
    package_config = ModelingConfig(
        model=model_config,
        alerts=alert_config,
        algorithm=algo_config,
        allow_change=allow_change
    )
    for path in config_path:
        with open(path, "w") as f:
            json.dump(package_config.model_dump(), f, indent=4)
