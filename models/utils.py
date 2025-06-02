import torch
from transformers import BitsAndBytesConfig


def bitsandbytes_8bit_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
    )
