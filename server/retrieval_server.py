import os
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_serializer

from services.retrieval_service import RetrievalService
from utils.image_utils import image_to_base64
from utils.utils import load_yaml

app = FastAPI(title="Retrieval Server")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    images: List[str]
    image_paths: List[str]
    scores: List[float]
    success: bool
    message: str


class ProcessFeedbackRequest(BaseModel):
    query: str
    relevant_image_paths: List[str]
    annotator_json_boxes_list: List[Any]
    visualization: bool = False
    top_k_feedback: int = 5
    prompt_based_on_query: bool = False
    relevant_captions: Optional[Union[List[str], str]] = None
    irrelevant_captions: Optional[Union[List[str], str]] = None
    prompt: Optional[str] = None


class ApplyFeedbackRequest(BaseModel):
    query: str
    top_k: int
    relevant_captions: Optional[List[str]] = None
    irrelevant_captions: Optional[List[str]] = None
    fuse_initial_query: bool = False


class ProcessFeedbackResponse(BaseModel):
    relevance_feedback_results: Dict[str, Any]
    success: bool
    message: str

    @field_serializer('relevance_feedback_results')
    def serialize_relevance_feedback_results(self, value):
        if isinstance(value, dict):
            serialized = {}
            for key, val in value.items():
                if isinstance(val, torch.Tensor):
                    serialized[key] = val.tolist()
                elif key == 'explanation' and val is not None:
                    if isinstance(val, list):
                        serialized[key] = [image_to_base64(img) for img in val]
                    else:
                        serialized[key] = image_to_base64(val)
                else:
                    serialized[key] = val
            return serialized
        return value

class ApplyFeedbackResponse(BaseModel):
    images: List[str]
    image_paths: List[str]
    scores: List[float]
    success: bool
    message: str


retrieval_service: Optional[RetrievalService] = None


@app.on_event("startup")
async def startup_event():
    global retrieval_service

    config_path = os.getenv("CONFIG_PATH", "configs/demo/coco_clip_large.yaml")
    captioning_config_path = os.getenv("CAPTIONING_CONFIG_PATH", "configs/captioning/llava_8bit.yaml")

    config = load_yaml(config_path)
    captioning_config = load_yaml(captioning_config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    retrieval_service = RetrievalService(
        config=config,
        captioning_model_config=captioning_config,
        device=device,
    )


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    try:
        images, scores, image_paths = retrieval_service.search_images(request.query, request.top_k)
        images = [image_to_base64(img) for img in images]
        return SearchResponse(
            images=images,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Search completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_feedback", response_model=ProcessFeedbackResponse)
async def process_feedback(request: ProcessFeedbackRequest):
    try:
        relevance_feedback_results = retrieval_service.process_feedback(
            query=request.query,
            relevant_image_paths=request.relevant_image_paths,
            annotator_json_boxes_list=request.annotator_json_boxes_list,
            visualization=request.visualization,
            top_k_feedback=request.top_k_feedback,
            prompt_based_on_query=request.prompt_based_on_query,
            relevant_captions=request.relevant_captions,
            irrelevant_captions=request.irrelevant_captions,
            prompt=request.prompt
        )
        return ProcessFeedbackResponse(
            relevance_feedback_results=relevance_feedback_results,
            success=True,
            message="Feedback processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply_feedback", response_model=ApplyFeedbackResponse)
async def apply_feedback(request: ApplyFeedbackRequest):
    try:
        images, scores, image_paths = retrieval_service.apply_feedback(
            query=request.query,
            top_k=request.top_k,
            relevant_captions=request.relevant_captions,
            irrelevant_captions=request.irrelevant_captions,
            fuse_initial_query=request.fuse_initial_query
        )
        images = [image_to_base64(img) for img in images]
        return ApplyFeedbackResponse(
            images=images,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Feedback applied successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
