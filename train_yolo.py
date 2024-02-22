import os

from clearml import Dataset, Task
from ultralytics import YOLO
from utils.clearml_utils import download_model, get_dataset_from_storage


def get_model_name_from_choice(model_name: str, model_variant: str) -> str:
    mapping = {
        ("YOLOv5", "small"): "yolov5s",
        ("YOLOv5", "medium"): "yolov5m",
        ("YOLOv5", "large"): "yolov5l",
        ("YOLOv8", "small"): "yolov8s",
        ("YOLOv8", "medium"): "yolov8m",
        ("YOLOv8", "large"): "yolov8l",
    }

    return mapping.get((model_name, model_variant), "")


# def get_dataset_from_storage(dataset_id: str) -> str:
#     """
#             ```
#         yolov5/
#             temp/
#                 {dataset_name}.zip
#             datasets/ <- NOTE: this is requried in ultraalytics config
#                 {dataset_name}/
#                     train/
#                     test/
#                     val/
#                     *.yaml -> return filepath of *.yaml
#         ```
#     """
#     from pathlib import Path

#     def _get_yaml_files(folder_path: str) -> list[Path]:
#         folder_path = Path(folder_path)
#         yaml_files = folder_path.glob("*.yaml")
#         return list(yaml_files)

#     def _check_zip_file(folder_path: str):
#         folder_path = Path(folder_path)
#         zip_files = list(folder_path.glob("*.zip"))
#         zip_file = None
#         if len(zip_files):
#             zip_file = zip_files[0]
#             os.system(f"unzip \"{zip_file}\" -d \"{folder_path}\"")

#         return zip_file

#     dataset = Dataset.get(dataset_id=dataset_id)

#     dataset_dir = os.path.join(os.getcwd(), "datasets")
#     folderpath = os.path.join(dataset_dir, dataset.name)
#     from ultralytics import settings
#     settings.update({'datasets_dir': dataset_dir})

#     os.makedirs(dataset_dir, exist_ok=True)
#     os.makedirs(folderpath, exist_ok=True)

#     dataset.get_mutable_local_copy(
#         target_folder=folderpath,
#         overwrite=True
#     )

#     zip_file = _check_zip_file(folderpath)
#     if zip_file:
#         print(f"Found zip file at path {zip_file}")

#     # Assumpe there only 1 *.yaml file
#     yaml_filepath: Path = _get_yaml_files(folderpath)[0]
#     yaml_filepath: str = str(yaml_filepath.absolute())

#     return yaml_filepath


def train_yolo(
        dataset_id: str,
        model_version: str = "yolov5s",
        batch_size: int = 16,
        imgsz: int = 640,
        epochs: int = 10,
        pretrained_model_id: str = None
) -> None:

    # yaml_filepath = get_dataset_zip_from_storage(dataset_id=dataset_id)
    yaml_filepath = get_dataset_from_storage(dataset_id=dataset_id)

    print(f"Dataset is stored at {yaml_filepath}")
    print("Complete prepared dataset, continue to training the model...")
    model_path = f"{model_version}.pt"
    if pretrained_model_id is not None:
        pretrained_model_path = download_model(model_id=pretrained_model_id)
        model_path = pretrained_model_path
    print("Model_path", model_path)
    model = YOLO(model_path)
    model.train(
        data=yaml_filepath,
        imgsz=imgsz,
        epochs=epochs,
        cache='ram',
        batch=batch_size
    )


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_id", default="yolov5s", help="ClearML dataset id"
    )
    args.add_argument(
        "--pretrained_model_id", default=None, help="ClearML pretained mopdel id"
    )
    args.add_argument(
        "--model_version", default="yolov5s", help="Model version"
    )
    args.add_argument(
        "--batch_size", default=16, type=int, help="Batch size"
    )
    args.add_argument(
        "--imgsz", default=640, help="Image size", type=int,
    )
    args.add_argument(
        "--epochs", default=10, help="Epochs", type=int,
    )

    args.add_argument(
        "--project", default="YOLOv5", help="ClearML Project Name",
    )
    args.add_argument(
        "--name", default="Train YOLOv5", help="ClearML Task name",
    )

    args = args.parse_args()

    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name=args.project,
            task_name=args.name,
        )

    train_yolo(
        dataset_id=args.dataset_id,
        model_version=args.model_version,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        epochs=args.epochs,
        pretrained_model_id=args.pretrained_model_id)
