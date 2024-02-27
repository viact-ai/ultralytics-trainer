from clearml import Model, Dataset
import os
from typing import List, Dict


def download_model(model_id: List[str]) -> Dict[str, str]:
    '''Download model from ClearML Registry'''
    model_paths = {}
    for id in model_id:
        print(f"Download model_id {id} from clearML Model Registry")
        model = Model(model_id=id)
        tmp_path = model.get_local_copy(extract_archive=True,
                                        force_download=True)
        # if not tmp_path:
        #     raise ValueError(
        #         "Could not download model, you must mistake InputModel & OutputModel")
        print(f"Model stored at {tmp_path}")
        model_paths[id] = tmp_path
    return model_paths


def get_dataset_from_storage(dataset_id: str) -> str:
    """
            ```
        yolov5/
            temp/
                {dataset_name}.zip
            datasets/ <- NOTE: this is requried in ultraalytics config
                {dataset_name}/
                    train/
                    test/
                    val/
                    *.yaml -> return filepath of *.yaml
        ```
    """
    from pathlib import Path

    def _get_yaml_files(folder_path: str) -> list[Path]:
        folder_path = Path(folder_path)
        yaml_files = folder_path.glob("*.yaml")
        return list(yaml_files)

    def _check_zip_file(folder_path: str):
        folder_path = Path(folder_path)
        zip_files = list(folder_path.glob("*.zip"))
        zip_file = None
        if len(zip_files):
            zip_file = zip_files[0]
            os.system(f"unzip \"{zip_file}\" -d \"{folder_path}\"")

        return zip_file

    dataset = Dataset.get(dataset_id=dataset_id)

    dataset_dir = os.path.join(os.getcwd(), "datasets")
    folderpath = os.path.join(dataset_dir, dataset.name)
    from ultralytics import settings
    settings.update({'datasets_dir': dataset_dir})

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(folderpath, exist_ok=True)

    dataset.get_mutable_local_copy(
        target_folder=folderpath,
        overwrite=True
    )

    zip_file = _check_zip_file(folderpath)
    if zip_file:
        print(f"Found zip file at path {zip_file}")

    # Assumpe there only 1 *.yaml file
    yaml_filepath: Path = _get_yaml_files(folderpath)[0]
    yaml_filepath: str = str(yaml_filepath.absolute())

    return yaml_filepath
