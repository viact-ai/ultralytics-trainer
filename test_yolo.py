import os
from ultralytics import YOLO
from clearml import Dataset
from clearml.task import Task


from utils.clearml_utils import download_model, get_dataset_from_storage
from utils.ultralytics import parse_metrics, report_metrics


def test_yolo(
    dataset_id: str,
    model_id: str,
    batch_size: int,
    imgsz: int,
) -> None:
    """
    Sample output: ultralytics.utils.metrics.DetMetrics object with attributes:
    ```
        ap_class_index: array([0])
        box: ultralytics.utils.metrics.Metric object
        confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x7fd3a588fac0>
        fitness: 0.12331301833487544
        keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        maps: array([    0.11055,     0.11055])
        names: {0: 'canvas', 1: 'person'}
        plot: True
        results_dict: {'metrics/precision(B)': 0.05601317957166392, 'metrics/recall(B)': 1.0, 'metrics/mAP50(B)': 0.23813998138422993, 'metrics/mAP50-95(B)': 0.11055446688494716, 'fitness': 0.12331301833487544}
        save_dir: PosixPath('runs/detect/val2')
        speed: {'preprocess': 0.11606216430664062, 'inference': 8.380484580993652, 'loss': 0.00152587890625, 'postprocess': 1.4193296432495117}
    ```
    """
    yaml_filepath = get_dataset_from_storage(dataset_id=dataset_id)

    print(f"Dataset is stored at {yaml_filepath}")
    print("Complete prepared dataset, continue to training the model...")

    model_path = download_model(model_id)
    model = YOLO(model_path)

    metrics = model.val(
        data=yaml_filepath,
        imgsz=imgsz,
        batch_size=batch_size
    )
    report_metrics(metrics)

    return metrics


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_id", default=None, help="ClearML dataset id",
    )
    args.add_argument(
        "--model_id", help="Model from ClearML Registry", type=str, default=None
    )
    args.add_argument(
        "--model_arch",
        help="Model type",
        type=str,
    )
    args.add_argument(
        "--batch_size", default=16, type=int, help="Batch size"
    )
    args.add_argument(
        "--imgsz", default=640, help="Image size", type=int,
    )

    args = args.parse_args()

    test_yolo(
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        imgsz=args.imgsz,
        batch_size=args.batch_size
    )
