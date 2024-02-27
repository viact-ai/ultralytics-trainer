import argparse

from typing import Dict, Any


from clearml import Task
from utils.clearml_utils import download_model
from utils.security import encrypt_model
from enums.config import ModuleType, get_value
import utils.export as export_utils


def parse_pairs(pairs):
    params = {}
    for pair in pairs:
        key, value = pair.split('=')
        params[key] = value
    return params


def parse_list(params):
    values = []
    for value in params:
        values.append(value.split(","))
    return values


def package_ops(
    module: str,
    version: str,
    model_infos: Dict[str, Dict[str, str]],
    module_config: Dict[str, Any] = None
) -> tuple[str, str]:
    '''
        Return: zip_filepath, zip_filepath
            full_path, filename
    '''
    module = get_value(ModuleType, module)
    if module is not None:
        # artifact_config = Path("./default_config.json")
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     temp_dir_path = Path(temp_dir)
        zip_filepath, zip_filename, artifact_config = export_utils.get_zipfile(module=module,
                                                                               version=version,
                                                                               model_infos=model_infos,
                                                                               module_config=module_config)

        return zip_filepath, zip_filename, artifact_config
    return None, None, None


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        "--model_id",
        help="Model from ClearML Registry",
        nargs="+",
        default=None,
        action="append")

    args_parser.add_argument(
        "--model_arch",
        help="Model from ClearML Registry",
        nargs="+",
        default=None,
        action="append")

    args_parser.add_argument(
        "--label_list",
        help="List of labels",
        nargs="+",
        default=None)

    args_parser.add_argument(
        "--version",
        help="Model version",
        default="0.0.0",
        type=str)

    args_parser.add_argument(
        "--encrypt",
        help="Encrypt model",
        type=int,
        choices=[0, 1],
        default=1,
    )
    args_parser.add_argument(
        "--imgsz",
        help="Image size of model",
        type=int,
        default=640,
    )
    args_parser.add_argument(
        "--ai_module",
        help="Module type",
        type=str,
        default=None,
    )
    args_parser.add_argument(
        "--module_config",
        help="Extend model config for model id for mapping function",
        nargs="+",
        default=None
    )
    args = args_parser.parse_args()

    labels = parse_list(args.label_list)

    if not args.model_id:
        raise ValueError("`model_id` must be provided")

    model_paths = download_model(args.model_id)

    model_infos = export_utils.export_to_onnx(model_paths=model_paths,
                                              labels=labels,
                                              model_archs=args.model_arch,
                                              imgsz=args.imgsz)

    if args.encrypt:
        # Encrypt model
        for model_id, info in model_infos.items():
            path = info["onnx_path"]
            encrypt_model(
                input_path=path,
                output_path=path,
            )
            print(f"Encrypted model stored at: {path}")

    zip_filepath, name, config_path = package_ops(
        version=args.version,
        module=args.ai_module,
        model_infos=model_infos,
        module_config=parse_pairs(args.module_config),
    )

    task = Task.current_task()
    if not task:
        task = Task.init(project_name="Package-engine-for-MLops",
                         task_name=f"Export module {args.ai_module} version {args.version}")
    if task:
        print(f"Found current task {task.id}")
        print(f"Uploading to clearML server")

        task.upload_artifact(
            name="exported_model_zip",
            artifact_object=zip_filepath,
        )

        # task.upload_artifact(
        #     name="exported_onnx_model",
        #     artifact_object=onnx_model_path,
        # )
        task.upload_artifact(
            name="default_config",
            artifact_object=config_path
        )
        print("Complete upload package to clearML server")
    else:
        print("Current clearML task not found, will not sync artifact to clearML")
