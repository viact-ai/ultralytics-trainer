import os
import sys
from pathlib import Path
import json

from enums.config import ModelingType, ModuleType
from schema.config import ModelingDefaultConfig

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # src root
WS_ROOT = FILE.parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

if str(WS_ROOT) not in sys.path:
    sys.path.append(str(WS_ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WS_ROOT = Path(os.path.relpath(WS_ROOT, Path.cwd()))

AVAILABLE_MODULES = [str(module.value) for module in ModuleType]


def load_default_config(module: str):
    config_path = None
    if module in AVAILABLE_MODULES:
        config_path = os.path.join(ROOT, "configs", "detection.json")
    else:
        print("Not support modeling type {}".format(module))

    default_config = json.load(open(config_path, 'r'))
    default_data = ModelingDefaultConfig(**default_config)
    # print(default_data.to_dict())
    return default_data
