import numpy as np
from PIL import Image

from services.retrieval_service import RetrievalServiceVisual


def _default_config():
    return {
        "IMAGE_CORPUS_PATH": "data/coco/",
        "INDEX_PATH": "faiss/coco/openai/clip-vit-large-patch14/image_index.faiss",
        "VLM_MODEL_FAMILY": "clip",
        "VLM_MODEL_NAME": "openai/clip-vit-large-patch14",
        "IMG_SIZE": 224,
        "PATCH_SIZE": 32,
    }


def _init_retrieval_service():
    return RetrievalServiceVisual(
        config=_default_config(),
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

    relevance_feedback_results = retrieval_service.process_and_apply_feedback(
        query="a photo of a cat",
        relevant_image_paths=image_paths,
        annotator_json_boxes_list=annotator_json_boxes_list,
        top_k=5,
        fuse_initial_query=True,
    )

    print(relevance_feedback_results)
