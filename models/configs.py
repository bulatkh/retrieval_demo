from typing import Any, Dict, Optional

from transformers import (
    AutoProcessor,
    CLIPModel,
    LlavaForConditionalGeneration,
    SiglipModel,
)

from models.clip import CLIPWrapper
from models.llava import LLaVaWrapper
from models.siglip import SigLipWrapper

# model_id defines the default model_id that can be overwritten
CONFIGS = {
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "model_class": CLIPModel,
        "processor_class": AutoProcessor,
        "wrapper_class": CLIPWrapper,
    },
    "siglip": {
        "model_id": "google/siglip-base-patch16-256",
        "model_class": SiglipModel,
        "processor_class": AutoProcessor,
        "wrapper_class": SigLipWrapper,
    },
    "llava": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_class": LlavaForConditionalGeneration,
        "processor_class": AutoProcessor,
        "wrapper_class": LLaVaWrapper,
    },
}


def get_model_config(
        model_family: str,
        model_id: Optional[str] = None,
) -> Dict[str, Any]:
    config = CONFIGS.get(model_family, {})
    if not config:
        raise ValueError(f"Model config is not parsed. Plase use model_family from {list(CONFIGS.keys())}")
    if model_id is not None:
        config["model_id"] = model_id
    return config
