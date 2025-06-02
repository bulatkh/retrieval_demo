from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from models.utils import bitsandbytes_8bit_config
from models.vlm_wrapper import VLMWrapperCaptioning


def init_llava(
        model_config: Dict[str, Any],
        device: str = "cuda",
        use_8bit: bool = False
    ):
    model = model_config["model_class"].from_pretrained(
        model_config["model_id"],
        quantization_config=bitsandbytes_8bit_config() if use_8bit else None
    )
    model = model.to(device) if not use_8bit else model
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])

    vlm_wrapper = model_config["wrapper_class"](model=model, processor=processor)
    return vlm_wrapper

@dataclass
class LLaVaWrapper(VLMWrapperCaptioning):
    model: Any = field(
        default_factory=lambda: LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            device_map={"": 0},
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf"
        )
    )

    def __post_init__(self):
        self.processor.tokenizer.padding_side = "left"

    def process_inputs(self, apply_template=True, **kwargs):
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")

        if apply_template:
            prompts = [
                f"USER: <image>\n{prompt} ASSISTANT:" for prompt in kwargs['prompt']
            ]
        else:
            prompts = kwargs['prompt']

        return self.processor(
            images=kwargs['image'],
            text=prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

    def decode(self, outputs, **kwargs):
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        clean_up_tokenization_spaces = kwargs.get('clean_up_tokenization_spaces', False)
        return self.processor.batch_decode(
            outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

    def generate(self, inputs: Dict[str, Any], **kwargs) -> Any:
        # max_len = kwargs.get('max_len', 1000)
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        return self.model.generate(**inputs, max_new_tokens=max_new_tokens)
