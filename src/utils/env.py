import os 
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional


load_dotenv(dotenv_path=Path(".env"), override=False)

_DEFAULTS = {
    "DATA_DIR": "./data",
    "RUNS_DIR": "./runs",
    "DEVICE": "cuda",
    "NUM_WORKERS": "4",
    "SEED": "42",
    "LOG_LEVEL": "INFO",
    "IMAGE_SIZE": "256",
    "BACKBONE": "resnet50",
}


def get_env(key: str) -> str: return os.getenv(key, _DEFAULTS.get(key))

def get_path(key: str) -> Path: return Path(get_env(key)).expanduser().resolve()


DATA_DIR = get_path("DATA_DIR")
RUNS_DIR = get_path("RUNS_DIR")
DEVICE = get_env("DEVICE")
NUM_WORKERS = int(get_env("NUM_WORKERS"))
SEED = int(get_env("SEED"))
LOG_LEVEL = get_env("LOG_LEVEL")
IMAGE_SIZE = int(get_env("IMAGE_SIZE"))
BACKBONE = get_env("BACKBONE")

if __name__ == "__main__":
    print("DATA_DIR:", DATA_DIR)
    print("RUNS_DIR:", RUNS_DIR)
    print("DEVICE:", DEVICE)
    print("NUM_WORKERS:", NUM_WORKERS)
    print("SEED:", SEED)
    print("LOG_LEVEL:", LOG_LEVEL)
    print("IMAGE_SIZE:", IMAGE_SIZE)
    print("BACKBONE:", BACKBONE)