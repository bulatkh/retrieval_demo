import numpy as np
from PIL import Image

from services.retrieval_service import RetrievalService


def _default_config():
    return {
        "IMAGE_CORPUS_PATH": "data/coco/",
        "INDEX_PATH": "faiss/coco/openai/clip-vit-large-patch14/image_index.faiss",
        "VLM_MODEL_FAMILY": "clip",
        "VLM_MODEL_NAME": "openai/clip-vit-large-patch14",
        "IMG_SIZE": 224,
        "PATCH_SIZE": 32,
    }


def _default_captioning_config():
    return {
        "MODEL_FAMILY": "llava",
        "MODEL_ID": "llava-hf/llava-1.5-7b-hf",
        "USE_8BIT": True,
        "PROMPT": "Describe distinct features of the image in 5-10 words.",
    }


def _init_retrieval_service():
    return RetrievalService(
        config=_default_config(),
        captioning_model_config=_default_captioning_config(),
        alpha=0.6,
        beta=0.2,
        gamma=0.2,
    )


def test_default_retrieval_service_init():
    retrieval_service = _init_retrieval_service()
    assert retrieval_service is not None


def test_search_images():
    retrieval_service = _init_retrieval_service()
    images, scores, image_paths = retrieval_service.search_images("a photo of a cat")
    assert images is not None
    assert scores is not None
    assert image_paths is not None
    assert len(images) == 5
    assert len(scores) == 5
    assert len(image_paths) == 5
    print(image_paths)


def test_process_feedback():
    retrieval_service = _init_retrieval_service()
    images, scores, image_paths = retrieval_service.search_images("a photo of a cat")
    assert images is not None
    assert scores is not None
    assert image_paths is not None
    assert len(images) == 5
    assert len(scores) == 5
    assert len(image_paths) == 5

    annotator_json_boxes_list = (
        [
            {'label': 'Relevant', 'color': [0, 255, 0], 'xmin': 52, 'ymin': 33, 'xmax': 192, 'ymax': 192},
        ],
        [
            {'label': 'Irrelevant', 'color': [255, 0, 0], 'xmin': 12, 'ymin': 41, 'xmax': 202, 'ymax': 118},
        ],
        [
            {'label': 'Relevant', 'color': [0, 255, 0], 'xmin': 42, 'ymin': 37, 'xmax': 210, 'ymax': 180},
        ],
        [
            {'label': 'Relevant', 'color': [0, 255, 0], 'xmin': 36, 'ymin': 47, 'xmax': 209, 'ymax': 193},
        ],
        [
            {'label': 'Irrelevant', 'color': [255, 0, 0], 'xmin': 19, 'ymin': 69, 'xmax': 152, 'ymax': 211}
        ] 
    )

    relevance_feedback_results = retrieval_service.process_feedback(
        query="a photo of a cat",
        relevant_image_paths=image_paths,
        annotator_json_boxes_list=annotator_json_boxes_list,
        top_k_feedback=5,
    )

    assert relevance_feedback_results
    assert relevance_feedback_results is not None
    assert "positive" in relevance_feedback_results
    assert "negative" in relevance_feedback_results
    assert "relevant_captions" in relevance_feedback_results
    assert "irrelevant_captions" in relevance_feedback_results
    assert "explanation" in relevance_feedback_results
    