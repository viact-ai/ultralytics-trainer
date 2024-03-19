import argparse

from typing import Dict, Any, List
import json


from clearml import Task
from utils.security import encrypt_model
from enums.config import ModuleType, get_value
import utils.export as export_utils
from schema.export import ModelInfo


def package_ops(
    module: str,
    version: str,
    model_infos: Dict[str, Dict[str, str]],
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
                                                                               model_infos=model_infos)

        return zip_filepath, zip_filename, artifact_config
    return None, None, None


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
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
        "--ai_module",
        help="Module type",
        type=str,
        default=None,
    )
    args_parser.add_argument(
        "--models",
        help="All models",
        type=str,
        default=None,
    )

    args = args_parser.parse_args()

    models = []
    if args.models is not None:
        model_dict: list = json.loads(args.models)
        models = [ModelInfo(**model) for model in model_dict]

    model_infos = export_utils.export_to_onnx(models=models)

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
