import json
import os
from datetime import datetime

import yaml


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
