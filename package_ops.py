import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Union
import os

from clearml import Task

from utils.clearml_utils import download_model
import utils.export as export_utils
from utils.security import encrypt_model
from enums.config import (ModuleType,
                          get_value)


def package_ops(
    module: str,
    imgsz: Union[int, tuple, list],
    model_arch: str,
    version: str,
    label_list: list[str],
    model_path: Path,
    model_class_names: dict = None
) -> tuple[str, str]:
    '''
        Return: zip_filepath, zip_filepath
            full_path, filename
    '''
    module = get_value(ModuleType, module)
    if module is not None:
        is_valid = export_utils.check_class(
            module, list(model_class_names.values()))
        if is_valid:
            artifact_config = Path("./default_config.json")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)

                output_dir = temp_dir_path / f"{model_arch}_{version}"

                configs_dir = output_dir / "configs"
                configs_dir.mkdir(parents=True, exist_ok=True)
                weights_dir = output_dir / "weights"
                weights_dir.mkdir(parents=True, exist_ok=True)
                labels_txt_path = output_dir / "labels.txt"
                config_path = configs_dir / "default_config.json"

                classes = export_utils.label_list_to_txt(label_list,
                                                         labels_txt_path,
                                                         model_class_names)
                if isinstance(imgsz, int):
                    imgsz = [imgsz, imgsz]
                elif isinstance(imgsz, tuple) or isinstance(imgsz, list):
                    imgsz = list(imgsz)

                export_utils.get_default_config(config_path=[config_path, artifact_config],
                                                imgsz=imgsz,
                                                model_arch=model_arch,
                                                module=module,
                                                classes=classes
                                                )
                onnx_model_filepath = model_path

                zip_filepath = Path(f"{model_arch}_{version}.zip")
                with zipfile.ZipFile(zip_filepath, "w") as zipf:
                    zipf.write(labels_txt_path, arcname="labels.txt")
                    zipf.write(
                        config_path, arcname=f"configs/default_config.json")
                    zipf.write(onnx_model_filepath,
                               arcname=f"weights/best.onnx")

                return str(zip_filepath.absolute()), str(zip_filepath), artifact_config
    return None, None, None


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        "--model_id", help="Model from ClearML Registry", type=str, default=None)
    args_parser.add_argument(
        "--model_path", help="Model from local storage", default=None, type=str)
    args_parser.add_argument(
        "--model_arch",
        help="Model type",
        default=None,
        type=str,
        choices=[
            "yolov5s",
            "yolov5m",
            "yolov5l",
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
        ],
    )
    args_parser.add_argument(
        "--label_list", help="List of labels", action="append")
    args_parser.add_argument(
        "--version", help="Model version", default="1.0.0", type=str)
    args_parser.add_argument(
        "--encrypt",
        help="Encrypt model",
        type=int,
        choices=[0, 1],
        default=0,
    )
    args_parser.add_argument(
        "--imgsz",
        help="Image size of model",
        type=int,
        default=0,
    )
    args_parser.add_argument(
        "--ai_module",
        help="Module type",
        type=str,
        default="danger-zone",
    )
    args = args_parser.parse_args()

    if not args.model_id and not args.model_path:
        raise ValueError("`model_id` or `model_path` must be provided")

    model_path = args.model_path or download_model(args.model_id)

    onnx_model_path, class_names = export_utils.export_to_onnx(model_path=model_path,
                                                               imgsz=args.imgsz)

    print(f"ONNX model stored at: {onnx_model_path}")

    print(f"ONNX model stored at: {onnx_model_path}")

    if args.encrypt:
        # Encrypt model
        encrypt_model(
            input_path=onnx_model_path,
            output_path=onnx_model_path,
        )
        print(f"Encrypted model stored at: {onnx_model_path}")

    zip_filepath, name, config_path = package_ops(
        model_arch=args.model_arch,
        version=args.version,
        label_list=args.label_list,
        model_path=onnx_model_path,
        imgsz=args.imgsz,
        module=args.ai_module,
        model_class_names=class_names
    )
    print(f"Package stored at: {name} {zip_filepath}")

    # to upload with clearML
    task = Task.current_task()
    if task:
        print(f"Found current task {task.id}")
        print(f"Uploading to clearML server")

        task.upload_artifact(
            name="exported_model_zip",
            artifact_object=zip_filepath,
        )

        task.upload_artifact(
            name="exported_onnx_model",
            artifact_object=onnx_model_path,
        )
        task.upload_artifact(
            name="default_config",
            artifact_object=config_path
        )
        print("Complete upload package to clearML server")
    else:
        print("Current clearML task not found, will not sync artifact to clearML")
    # print(config_path)
    # os.system(f"rm -rf {config_path}")
