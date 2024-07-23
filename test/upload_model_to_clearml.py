from clearml import OutputModel, Task


def main():
    task = Task.init(
        project_name="local_dev",
        task_name="upload model",
    )

    model_path = "./models/unauthorized_access_by_vest/classify.pt"

    # Register and upload the model to ClearML
    output_model = OutputModel(task=task, framework='Pytorch')
    output_model.update_weights(weights_filename=model_path, upload_uri="s3://viact-mlops")
    output_model.publish()

    print(f"Model registered and uploaded to ClearML: {output_model.id}")


if __name__ == "__main__":
    main()
