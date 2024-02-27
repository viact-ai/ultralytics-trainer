from ultralytics import YOLO
from pathlib import Path
import json
import tempfile
import zipfile

from typing import Dict, List, Any
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
    get_value,
    BaseClasses,
    ModuleType,
    YOLOTasks,
    MAPPING_MODULE_TO_MODELING,
    MAPPING_YOLO_TASK_TO_MODELING,
    MODULE_CLASSES,
    DEFAULT_ALERT_STRING)


DEFAULT_OPSET = 12
DEFAULT_DYNAMIC = True


def label_list_to_txt(label_list: List[str],
                      labels_txt_path: Path,
                      model_class_names: dict = None) -> bool:
    flag = False
    names = list(model_class_names.values())
    if len(label_list) == 0:
        flag = True
    else:
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


def export_to_onnx(model_paths: Dict[str, str],
                   model_archs: List[str],
                   labels: List[List[str]] = None,
                   format: str = 'onnx',
                   imgsz: int = 640) -> Dict[str, Dict[str, str]]:
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
    model_infos = {}
    for i, (model_id, model_path) in enumerate(model_paths.items()):
        model = YOLO(model=model_path)
        img_size = imgsz
        if model.task == "classify":
            img_size = 224
        output_name = model.export(
            format=format,
            imgsz=img_size,
            dynamic=DEFAULT_DYNAMIC,
            opset=DEFAULT_OPSET
        )
        names = model.names
        print(f"ONNX model stored at: {output_name}")

        model_arch = None
        model_size = None

        model_arch = model_archs[i]
        if "yolov5" in model_arch:
            model_arch = "yolov5"
            model_size = model_arch.split("yolov5")[-1]
        elif "yolov8" in model_arch:
            model_arch = "yolov8"
            model_size = model_arch.split("yolov8")[-1]

        if isinstance(img_size, int):
            img_size = [img_size, img_size]

        model_infos[model_id] = {
            "onnx_path": output_name,
            "names": names,
            "labels": labels[i] if labels is not None else [],
            "model_arch": model_arch,
            "model_size": model_size,
            "task": model.task,
            "imgsz": img_size
        }
    return model_infos


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
                classes: List = None):
    if len(classes) == 0:
        return True
    count = 0
    neccessary_classes = MODULE_CLASSES[ai_module]
    if isinstance(neccessary_classes, list) \
            and len(neccessary_classes) == 1:
        for cls in neccessary_classes:
            if cls in classes:
                count += 1
        return count == len(neccessary_classes)
    elif isinstance(neccessary_classes, dict):
        flag = False
        for _, value in neccessary_classes.items():
            count = 0
            for cls in value:
                if cls in classes:
                    count += 1
            if count == len(value):
                flag = True
                continue
        return flag

    return False


def get_module_configs(ai_module: ModuleType,
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
        algo_config = PersonNearFenceParam(
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


def get_model_config(model_info: Dict[str, str],
                     weight_path: str,
                     label_path: str) -> ModelConfig:

    yolo_task = get_value(YOLOTasks, model_info["task"])
    modeling_type = MAPPING_YOLO_TASK_TO_MODELING[yolo_task]

    inference_config = InferenceConfig(
        imgsz=model_info["imgsz"],
        classes=None
    )
    # Update model config
    model_config = ModelConfig(
        arch=model_info["model_arch"],
        type=modeling_type,
        inference=inference_config,
        size=model_info["model_size"],
        weight_path=weight_path,
        label_path=label_path
    )
    return model_config


def get_default_config(
        config_path: list[Path],
        model_info: Dict[str, Dict[str, str]],
        module: ModuleType = None,
        module_config: Dict[str, str] = None,
        classes: list = None,
        **kwargs):
    # default_config = load_default_config(module=module)
    # model_arch_version = None
    # model_size = None
    # if "yolov5" in model_arch:
    #     model_arch_version = "yolov5"
    #     model_size = model_arch.split("yolov5")[-1]
    # elif "yolov8" in model_arch:
    #     model_arch_version = "yolov8"
    #     model_size = model_arch.split("yolov8")[-1]

    # inference_config = InferenceConfig(
    #     imgsz=imgsz,
    #     classes=None
    # )
    # modeling_type = MAPPING_MODULE_TO_MODELING[module]

    # Update model config
    models = [get_model_config(model_info[model_id])
              for model_id in model_info]
    allow_change = AllowChange(
        inference=["conf_threshold", "iou_threshold"],
        algorithm=get_algorithm_allow_change(ai_module=module)
    )
    if module_config is not None:
        if module == ModuleType.LIFTING_LOAD_DANGER_ZONE \
                and module_config["version"] == "v2":
            detect_model_id = module_config["detect"]
            classes = model_info[detect_model_id]["names"].values()
        else:
            model_ids = list(model_info.keys())
            classes = model_info[model_ids[0]]["names"].values()

    algo_config, alert_config = get_module_configs(
        ai_module=module, classes=classes)
    package_config = ModelingConfig(
        model=models,
        alerts=alert_config,
        algorithm=algo_config,
        allow_change=allow_change
    )
    for path in config_path:
        with open(path, "w") as f:
            json.dump(package_config.model_dump(), f, indent=4)


def get_zipfile(module: str,
                version: str,
                model_infos: Dict[str, Dict[str, str]],
                module_config: Dict[str, Any] = None):
    zip_filepath = Path(f"{module}_{version}.zip")
    models = []
    main_classes = []
    artifact_config_path = "./default_config.json"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_dir = temp_dir_path / f"{module}_{version}"

        configs_dir = output_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        for i, (model_id, model_info) in enumerate(model_infos.items()):
            is_valid = check_class(ai_module=module,
                                   classes=model_info["labels"])
            main_model_id = module_config["main_model_id"]
            model_task = model_info["task"]
            model_path = f"weights/model_{model_task}_{i}.onnx"
            label_path = f"labels_{i}.txt"

            labels_txt_path = output_dir / label_path
            # labels_txt_path.mkdir(parents=True, exist_ok=True)

            if is_valid:
                classes = label_list_to_txt(label_list=model_info["labels"],
                                            labels_txt_path=labels_txt_path,
                                            model_class_names=model_info["names"])

                model_config = get_model_config(model_info,
                                                weight_path=model_path,
                                                label_path=label_path)
                models.append(model_config)

                if model_id == main_model_id:
                    main_classes = classes

                onnx_model_filepath = model_info["onnx_path"]
                with zipfile.ZipFile(zip_filepath, "a") as zipf:
                    zipf.write(labels_txt_path,
                               arcname=label_path)
                    zipf.write(onnx_model_filepath,
                               arcname=model_path)

        allow_change = AllowChange(
            inference=["conf_threshold", "iou_threshold"],
            algorithm=get_algorithm_allow_change(ai_module=module)
        )

        algo_config, alert_config = get_module_configs(
            ai_module=module, classes=main_classes)
        package_config = ModelingConfig(
            model=models,
            alerts=alert_config,
            algorithm=algo_config,
            allow_change=allow_change
        )

        with open(artifact_config_path, "w") as f:
            json.dump(package_config.model_dump(), f, indent=4)

        with zipfile.ZipFile(zip_filepath, "a") as zipf:
            zipf.write(artifact_config_path,
                       arcname="configs/default_config.json")

    return str(zip_filepath.absolute()), str(zip_filepath), artifact_config_path
