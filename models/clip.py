from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from transformers import CLIPModel, CLIPProcessor

from models.vlm_wrapper import VLMWrapperRetrieval


@dataclass
class CLIPWrapper(VLMWrapperRetrieval):
    model: Any = field(
        default_factory=lambda: CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            device_map={"": 0},
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    )

    def process_inputs(self, images=None, text=None) -> Dict[str, Any]:
        assert images is not None or text is not None
        return self.processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

    def get_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        outputs = self.model(**inputs)
        return {
            'image_embeds': outputs.image_embeds,
            'text_embeds': outputs.text_embeds,
            'logits_per_image': outputs.logits_per_image,
            'logits_per_text': outputs.logits_per_text,
            'vision_model_output': outputs.vision_model_output.last_hidden_state,
            'text_model_output': outputs.text_model_output.last_hidden_state
        }

    def get_text_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        return self.model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    def get_image_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        return self.model.get_image_features(
            pixel_values=inputs['pixel_values']
        )
