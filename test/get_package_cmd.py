import json


def main():
    version = "0.0.0"
    ai_module = "lifting-load-danger-zone"

    names = []
    data = {}

    if ai_module == "ppe-detection" or ai_module == "safety-vest" or ai_module == "safety-helmet":
        names = [
            "person",
            "shoe",
            "no_shoe",
            "vest",
            "no_vest",
            "helmet",
            "no_helmet",
        ]

        data = {
            "model_id": "dcdfc64507bb4b108fece3aff9803f50",
            "model_arch": "yolov8n",
            "model_type": "DETECTION",
            "imgsz": 640,
            "label_list": names,
        }

    elif ai_module == "illegal-parking":
        names = [
            "truck",
        ]

        data = {
            "model_id": "46e5b045eba4433aa06c34e62a5ed19b",
            "model_arch": "yolov8n",
            "model_type": "DETECTION",
            "imgsz": 640,
            "label_list": names,
        }

    elif ai_module == "open-edge":
        names = [
            "person",
            "canvas",
            "hole",
            "fence",
        ]

        data = {
            "model_id": "2a3417539588463eb8f0d83620212834",
            "model_arch": "yolov8n",
            "model_type": "SEGMENTATION",
            "imgsz": 640,
            "label_list": names,
        }

    elif ai_module == "no-covering-shoes":
        names = [
            "person",
            "shoe",
            "no_shoe",
        ]

        data = {
            "model_id": "5b469c87dfbc4a3599ce5848d5dc3a86",
            "model_arch": "yolov8n",
            "model_type": "DETECTION",
            "imgsz": 640,
            "label_list": names,
        }
    elif ai_module == "anti-collision":
        names = [
            "car",
            "van",
            "bus",
            "truck",
            "motorcycle",
            "person",
            "face",
            "wheel",
            "excavator",
            "concrete_mixer",
            "bulldozer/backhoe/loader",
            "roller",
            "boom_lift",
            "scissor_lift",
            "mobile_crane",
            "machinery",
        ]

        data = {
            "model_id": "fc225c9bdd214db0ba386bf0ea31e880",
            "model_arch": "yolov8n",
            "model_type": "DETECTION",
            "imgsz": 640,
            "label_list": names,
        }
    elif ai_module == "outside-walking":
        names_1 = [
            "person",
        ]
        model_1 = {
            "model_id": "fb9a40ce346e4baebc3341920b30f02a",
            "model_arch": "yolov8n",
            "model_type": "DETECTION",
            "imgsz": 640,
            "label_list": names_1,
        }
        names_2 = [
            "orange",
            "others",
        ]
        model_2 = {
            "model_id": "2d3db53648fc4e08be1c3043588db8d3",
            "model_arch": "yolov8n",
            "model_type": "CLASSIFICATION",
            "imgsz": 640,
            "label_list": names_2,
        }
        data = [model_1, model_2]
    elif ai_module == "unauthorized-access":
        names_1 = [
            "person",
        ]
        model_1 = {
            "model_id": "699e08b0163d4b49bcfb2f591cf354df",
            "model_arch": "yolov8n",
            "model_type": "DETECTION",
            "imgsz": 640,
            "label_list": names_1,
        }
        names_2 = [
            "black",
            "blue",
            "green",
            "orange",
            "red",
            "underfine_img",
        ]
        model_2 = {
            "model_id": "26e6db6718b24113b5d752350e7d5e36",
            "model_arch": "yolov8n",
            "model_type": "CLASSIFICATION",
            "imgsz": 640,
            "label_list": names_2,
        }
        data = [model_1, model_2]
    elif ai_module == "lifting-load-danger-zone":
        names_1 = [
            "person",
            "hook",
        ]
        model_1 = {
            "model_id": "018d0c5147dc44bea0e064f0864a10de",
            "model_arch": "yolov8n",
            "model_type": "DETECTION",
            "imgsz": 640,
            "label_list": names_1,
        }
        names_2 = [
            "0",
            "1",
        ]
        model_2 = {
            "model_id": "59a75934026141cb9a8471748d73a925",
            "model_arch": "yolov8n",
            "model_type": "CLASSIFICATION",
            "imgsz": 640,
            "label_list": names_2,
        }
        data = [model_1, model_2]

    if isinstance(data, dict):
        escaped_str = json.dumps([data])
    else:
        escaped_str = json.dumps(data)
    escaped_str = escaped_str.replace('"', '\\"')

    cmd = ["python", "package_ops.py", "--encrypt", "1", "--version", version, "--ai_module", ai_module, "--models", f"'{escaped_str}'"]
    print(" ".join(cmd))


if __name__ == "__main__":
    main()
