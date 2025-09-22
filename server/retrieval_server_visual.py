import os
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.retrieval_service import RetrievalServiceVisual
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


class ProcessApplyFeedbackRequest(BaseModel):
    query: str
    top_k: int
    relevant_image_paths: List[str]
    annotator_json_boxes_list: List[Any]
    fuse_initial_query: bool = False


class ProcessApplyFeedbackResponse(BaseModel):
    images: List[str]
    image_paths: List[str]
    scores: List[float]
    success: bool
    message: str


retrieval_service: Optional[RetrievalServiceVisual] = None


@app.on_event("startup")
async def startup_event():
    global retrieval_service

    config_path = os.getenv("CONFIG_PATH", "configs/demo/coco_clip_large.yaml")

    config = load_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    retrieval_service = RetrievalServiceVisual(
        config=config,
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


@app.post("/apply_feedback", response_model=ProcessApplyFeedbackResponse)
async def apply_feedback(request: ProcessApplyFeedbackRequest):
    try:
        images, scores, image_paths = retrieval_service.process_and_apply_feedback(
            query=request.query,
            top_k=request.top_k,
            relevant_image_paths=request.relevant_image_paths,
            annotator_json_boxes_list=request.annotator_json_boxes_list,
            fuse_initial_query=request.fuse_initial_query
        )
        images = [image_to_base64(img) for img in images]
        return ProcessApplyFeedbackResponse(
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
