import torch

from models.attentive_summarizer import init_summarizer
from models.configs import get_model_config
from models.relevance_feedback import AFSRelevanceFeedback

AFS_CONFIG = {
    "pooler_config": {
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "depth": 2
    },
    "global_embeddings_vision": False,
    "global_embeddings_text": False,
    "no_caption_loss": True,
    "text_dim": 512,
    "vision_dim": 768,
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "batch_size": 512,
    "max_epochs": 100
}

CLIP_SUMMARIZER_CHECKPOINT = "checkpoints/clip-vit-base-patch32-2025-03-19_18_49_57_013812/epoch=25-val_loss=0.08.ckpt"

QUERY = "A man wearing a red shirt and a red helmet is sitting on a motorcycle."

TOP5_IMAGE_PATHS = [
    "data/coco/train2014/COCO_train2014_000000011690.jpg",
    "data/coco/train2014/COCO_train2014_000000008238.jpg",
    "data/coco/train2014/COCO_train2014_000000054286.jpg",
    "data/coco/test2014/COCO_test2014_000000110880.jpg",
    "data/coco/test2014/COCO_test2014_000000551964.jpg"
]


def _init_vlm_wrapper(model_family, model_id):
    config = get_model_config(model_family, model_id)
    processor = config["processor_class"].from_pretrained(config["model_id"])
    model = config["model_class"].from_pretrained(config["model_id"])
    model.eval()
    wrapper = config["wrapper_class"](model=model, processor=processor)
    return wrapper


def _init_summarizer_clip(config, summarizer_checkpoint=None):
    return init_summarizer(
        config,
        summarizer_checkpoint=summarizer_checkpoint
    )

def test_afsrf_default():
    model_family = "clip"
    model_id = "openai/clip-vit-base-patch32"
    vlm_wrapper = _init_vlm_wrapper(model_family, model_id)
    summarizer = _init_summarizer_clip(AFS_CONFIG, CLIP_SUMMARIZER_CHECKPOINT)    
    afsrf = AFSRelevanceFeedback(
        model_family,
        vlm_wrapper,
        summarizer,
        img_size=vlm_wrapper.model.config.vision_config.image_size,
        patch_size=vlm_wrapper.model.config.vision_config.patch_size
    )

    with torch.no_grad():
        updated_query_embeddings, images_with_saliency = afsrf(
            QUERY,
            TOP5_IMAGE_PATHS
        )

    assert updated_query_embeddings.shape == vlm_wrapper.get_text_embeddings(
        vlm_wrapper.process_inputs(text=QUERY)
    ).shape

    assert images_with_saliency is None


def test_afsrf_visualization():
    model_family = "clip"
    model_id = "openai/clip-vit-base-patch32"
    vlm_wrapper = _init_vlm_wrapper(model_family, model_id)
    summarizer = _init_summarizer_clip(AFS_CONFIG, CLIP_SUMMARIZER_CHECKPOINT)    
    afsrf = AFSRelevanceFeedback(
        model_family,
        vlm_wrapper,
        summarizer,
        img_size=vlm_wrapper.model.config.vision_config.image_size,
        patch_size=vlm_wrapper.model.config.vision_config.patch_size
    )

    with torch.no_grad():
        updated_query_embeddings, images_with_saliency = afsrf(
            QUERY,
            TOP5_IMAGE_PATHS,
            visualization=True
        )

    assert updated_query_embeddings.shape == vlm_wrapper.get_text_embeddings(
        vlm_wrapper.process_inputs(text=QUERY)
    ).shape

    assert images_with_saliency is not None
    assert len(images_with_saliency) == len(TOP5_IMAGE_PATHS)