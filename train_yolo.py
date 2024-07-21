import os

from clearml import Dataset, Task
from ultralytics import YOLO

from enums.config import YOLOTasks
from utils.clearml_utils import download_model, get_dataset_from_storage


def get_model_name_from_choice(model_name: str, model_variant: str) -> str:
    mapping = {
        ("YOLOv5", "nano"): "yolov5n",
        ("YOLOv5", "small"): "yolov5s",
        ("YOLOv5", "medium"): "yolov5m",
        ("YOLOv5", "large"): "yolov5l",
        ("YOLOv5", "extra_large"): "yolov5x",
        ("YOLOv8", "nano"): "yolov8n",
        ("YOLOv8", "small"): "yolov8s",
        ("YOLOv8", "medium"): "yolov8m",
        ("YOLOv8", "large"): "yolov8l",
        ("YOLOv8", "extra_large"): "yolov8x",
        ("YOLOv10", "nano"): "yolov10n",
        ("YOLOv10", "small"): "yolov10s",
        ("YOLOv10", "medium"): "yolov10m",
        ("YOLOv10", "large"): "yolov10l",
        ("YOLOv10", "extra_large"): "yolov10x",
    }

    return mapping.get((model_name, model_variant), "")


def get_model_path(model_version: str, model_type: str):
    suffix = ""
    model_path = f"{model_version}.pt"
    if model_type == str(YOLOTasks.CLASSIFY):
        suffix = "cls"
    elif model_type == str(YOLOTasks.SEGMENT):
        suffix = "seg"

    if suffix != "":
        model_path = f"{model_version}-{suffix}.pt"

    return model_path


def train_yolo(
    dataset_id: str,
    model_version: str = "yolov8s",
    batch_size: int = 16,
    imgsz: int = 640,
    epochs: int = 50,
    pretrained_model_id: str = None,
    model_type: str = "detect",
    **kwargs,
) -> None:

    # yaml_filepath = get_dataset_zip_from_storage(dataset_id=dataset_id)
    dataset_filepath = get_dataset_from_storage(dataset_id=dataset_id)

    print(f"Dataset is stored at {dataset_filepath}")
    print("Complete prepared dataset, continue to training the model...")
    model_path = get_model_path(model_version=model_version, model_type=model_type)
    if pretrained_model_id is not None:
        pretrained_model_path = download_model(model_id=pretrained_model_id)
        model_path = pretrained_model_path
    print("Model_path", model_path)
    model = YOLO(model_path)
    model.train(
        data=dataset_filepath,
        imgsz=imgsz,
        epochs=epochs,
        cache="ram",
        batch=batch_size,
        cache=False,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--dataset_id", default="yolov5s", help="ClearML dataset id")
    args.add_argument(
        "--pretrained_model_id", default=None, help="ClearML pretained mopdel id"
    )
    args.add_argument("--model_version", default="yolov5s", help="Model version")
    args.add_argument("--batch_size", default=16, type=int, help="Batch size")
    args.add_argument(
        "--imgsz",
        default=640,
        help="Image size",
        type=int,
    )
    args.add_argument(
        "--epochs",
        default=10,
        help="Epochs",
        type=int,
    )
    args.add_argument("--model_type", default="detect", help="Task of model", type=str)
    args.add_argument(
        "--optimizer",
        default="auto",
        help=(
            "Choice of optimizer for training. Options include SGD, Adam,"
            " AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection"
            " based on model configuration. Affects convergence speed and stability."
        ),
    )
    args.add_argument(
        "--amp",
        default=False,
        help=(
            "Enables Automatic Mixed Precision (AMP) training,"
            " reducing memory usage and possibly speeding up"
            " training with minimal impact on accuracy."
        ),
    )
    args.add_argument(
        "--single-cls",
        default=False,
        help=(
            "Treats all classes in multi-class datasets as a"
            " single class during training. Useful for binary"
            " classification tasks or when focusing on object"
            " presence rather than classification."
        ),
        type=bool,
    )
    args.add_argument(
        "--cos-lr",
        default=False,
        help=(
            "Utilizes a cosine learning rate scheduler,"
            " adjusting the learning rate following a cosine"
            " curve over epochs. Helps in managing learning rate"
            " for better convergence."
        ),
    )
    args.add_argument(
        "--lr0",
        default=0.01,
        help=(
            "Initial learning rate (i.e. SGD=1E-2, Adam=1E-3)."
            " Adjusting this value is crucial for the optimization"
            " process, influencing how rapidly model weights are updated."
        ),
        type=float,
    )
    args.add_argument(
        "--lrf",
        default=0.1,
        help=(
            "Initial learning rate (i.e. SGD=1E-2, Adam=1E-3)."
            " Adjusting this value is crucial for the optimization"
            " process, influencing how rapidly model weights are updated."
        ),
        type=float,
    )
    args.add_argument(
        "--momentum",
        default=0.937,
        help=(
            "Momentum factor for SGD or beta1 for Adam optimizers,"
            " influencing the incorporation of past gradients in the current update."
        ),
    )
    args.add_argument(
        "--weight_decay",
        default=0.0005,
        help="L2 regularization term, penalizing large weights to prevent overfitting.",
    )
    args.add_argument(
        "--warmup_epochs",
        default=3,
        help=(
            "Number of epochs for learning rate warmup, gradually"
            " increasing the learning rate from a low value to the"
            " initial learning rate to stabilize training early on."
        ),
        type=int,
    )
    args.add_argument(
        "--warmup_momentum",
        default=0.8,
        help=(
            "Initial momentum for warmup phase, gradually adjusting"
            " to the set momentum over the warmup period."
        ),
    )
    args.add_argument(
        "--box",
        default=7.5,
        help=(
            "Weight of the box loss component in the loss function,"
            " influencing how much emphasis is placed on accurately"
            " predicting bounding box coordinates."
        ),
    )
    args.add_argument(
        "--cls",
        default=0.5,
        help=(
            "Weight of the classification loss in the total loss function,"
            " affecting the importance of correct class prediction relative to other components."
        ),
    )
    args.add_argument(
        "--dropout",
        default=0.0,
        help=(
            "Dropout rate for regularization in classification tasks,"
            " preventing overfitting by randomly omitting units during training."
        ),
    )

    args = args.parse_args()

    task = Task.current_task()
    if task is None:
        task_name = f"Train {args.model_version} {args.model_type} "
        task = Task.init(
            project_name=args.model_version,
            task_name=task_name,
        )

    train_yolo(
        dataset_id=args.dataset_id,
        model_version=args.model_version,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        epochs=args.epochs,
        pretrained_model_id=args.pretrained_model_id,
        model_type=args.model_type,
        single_cls=args.single_cls,
        cos_lr=args.cos_lr,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        box=args.box,
        cls=args.cls,
        dropout=args.dropout,
    )
