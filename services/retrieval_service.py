import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

import faiss
from models.configs import get_model_config
from models.llava import init_llava
from models.relevance_feedback import (
    CaptionVLMRelevanceFeedback,
    ImageBasedVLMRelevanceFeedback,
    RocchioUpdate, 
)
from utils.image_utils import resize_images


class RetrievalService:
    def __init__(
        self,
        config: Dict[str, Any],
        captioning_model_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        multiple: bool = False,
    ):
        self.config = config
        self.captioning_model_config = captioning_model_config if captioning_model_config is not None else None
        self.faiss_index = config["INDEX_PATH"]
        self.accumulated_query_embeddings = {"query_embedding": None}
        self.retrieval_round = 1
        self.experiment_id = 0
        self.device = device
        
        self._init_backbone()
        if self.captioning_model_config is not None:
            self._init_captioning_model()
            self._init_captioning_relevance_feedback()
        self._init_rocchio_update(alpha=alpha, beta=beta, gamma=gamma, multiple=multiple)
        self._init_faiss_index()

    def _init_backbone(self):
        self.backbone_config = get_model_config(
            self.config["VLM_MODEL_FAMILY"],
            self.config["VLM_MODEL_NAME"]
        )
        self.backbone = self.backbone_config["model_class"].from_pretrained(self.config["VLM_MODEL_NAME"])
        self.backbone.eval()
        self.backbone_processor = (
            self.backbone_config["processor_class"]
            .from_pretrained(self.config["VLM_MODEL_NAME"])
        )

        self.wrapper = self.backbone_config["wrapper_class"](
            model=self.backbone,
            processor=self.backbone_processor
        )
    
    def _init_captioning_model(self):
        model_config = get_model_config(
            self.captioning_model_config["MODEL_FAMILY"], 
            self.captioning_model_config["MODEL_ID"]
        )
        if self.captioning_model_config["MODEL_FAMILY"] == "llava":
            self.captioning_model = init_llava(
                model_config=model_config,
                device=self.device,
                use_8bit=self.captioning_model_config["USE_8BIT"]
            )
        else:
            raise ValueError(
                f"Captioning model family {self.captioning_model_config['model_family']} not supported"
            )

    def _init_captioning_relevance_feedback(self):
        self.captioning_relevance_feedback = CaptionVLMRelevanceFeedback(
            vlm_wrapper_retrieval=self.wrapper,
            vlm_wrapper_captioning=self.captioning_model,
        )

    def _init_rocchio_update(
        self,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        multiple: bool = False,
    ):
        self.rocchio_update = RocchioUpdate(alpha=alpha, beta=beta, gamma=gamma, multiple=multiple)

    def _init_faiss_index(self):
        try:
            self.index = faiss.read_index(self.faiss_index)
        except RuntimeError as e:
            raise ValueError(f"Failed to read FAISS index: {e}. Check if the index file exists.")
        try:
            with open(
                os.path.join(os.path.dirname(self.faiss_index),
                "image_paths.txt"),
                "r"
            ) as f:
                self.candidate_image_paths = [line.strip() for line in f.readlines()]
        except FileNotFoundError as e:
            raise ValueError(f"Failed to read image paths: {e}. Check if the image paths file exists.")

    def search_images(self, query: str, top_k: int = 5):
        """Extract image_search function logic"""
        self.experiment_id += 1

        processed_query = self.wrapper.process_inputs(text=query)
        with torch.no_grad():
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

        self.accumulated_query_embeddings["query_embedding"] = query_embedding

        scores, img_ids = self.index.search(query_embedding, top_k)
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        retrieved_images = resize_images(retrieved_images, self.config)

        return retrieved_images, scores, retrieved_image_paths

    def process_feedback(
        self,
        query: str,
        relevant_image_paths: List[str],
        user_prompt: Optional[str] = None,
        annotator_json_boxes_list: Optional[List[Any]] = None,
        visualization: bool = False,
        top_k_feedback: int = 5,
        prompt_based_on_query: bool = False,
        relevant_captions: Optional[Union[List[str], str]] = None,
        irrelevant_captions: Optional[Union[List[str], str]] = None,
        prompt: Optional[str] = None
    ):
        relevance_feedback_results = self.captioning_relevance_feedback(
            query=query,
            relevant_image_paths=relevant_image_paths,
            user_prompt=user_prompt,
            visualization=visualization,
            top_k_feedback=top_k_feedback,
            annotator_json_boxes_list=annotator_json_boxes_list,
            prompt_based_on_query=prompt_based_on_query,
            relevant_captions=relevant_captions,
            irrelevant_captions=irrelevant_captions,
            prompt=prompt
        )

        return {
            "positive": relevance_feedback_results["positive"].tolist() if relevance_feedback_results["positive"] is not None else None,
            "negative": relevance_feedback_results["negative"].tolist() if relevance_feedback_results["negative"] is not None else None,
            "relevant_captions": relevance_feedback_results["relevant_captions"],
            "irrelevant_captions": relevance_feedback_results["irrelevant_captions"],
            "explanation": relevance_feedback_results["explanation"]
        }

    def apply_feedback(
        self,
        query: str,
        top_k: int,
        relevant_captions: Optional[Union[List[str], torch.Tensor]] = None,
        irrelevant_captions: Optional[Union[List[str], torch.Tensor]] = None,
        fuse_initial_query: bool = False
    ):
        """Extract feedback_loop function logic"""
        processed_query = self.wrapper.process_inputs(text=query)
        with torch.no_grad():
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

        rocchio_query_embedding = (self.accumulated_query_embeddings["query_embedding"] + query_embedding) / 2 if (
            fuse_initial_query
        ) else self.accumulated_query_embeddings["query_embedding"]

        relevant_captions = [cap for cap in relevant_captions if cap != ""]
        irrelevant_captions = [cap for cap in irrelevant_captions if cap != ""]

        print(relevant_captions, irrelevant_captions)

        with torch.no_grad():
            if relevant_captions is not None and relevant_captions:
                positive_embeddings = self.wrapper.get_text_embeddings(
                        self.wrapper.process_inputs(text=relevant_captions)
                    )
                positive_embeddings = positive_embeddings.mean(dim=0)
            else:
                positive_embeddings = None
            if irrelevant_captions is not None and irrelevant_captions:
                negative_embeddings = self.wrapper.get_text_embeddings(
                self.wrapper.process_inputs(text=irrelevant_captions)
            )
                negative_embeddings = negative_embeddings.mean(dim=0)
            else:
                negative_embeddings = None

        self.accumulated_query_embeddings["query_embedding"] = self.rocchio_update(
            query_embeddings=rocchio_query_embedding,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings
        )

        scores, img_ids = self.index.search(self.accumulated_query_embeddings["query_embedding"], top_k)
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        retrieved_images = resize_images(retrieved_images, self.config)

        self.retrieval_round += 1
        return retrieved_images, scores, retrieved_image_paths


class RetrievalServiceVisual(RetrievalService):
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        multiple: bool = True,
    ):
        super().__init__(
            config=config,
            device=device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            multiple=multiple
        )
        self._init_image_based_relevance_feedback()

    def _init_image_based_relevance_feedback(self):
        self.image_based_relevance_feedback = ImageBasedVLMRelevanceFeedback(
            vlm_wrapper_retrieval=self.wrapper,
        )

    def process_and_apply_feedback(
        self,
        query: str,
        top_k: int,
        relevant_image_paths: List[str],
        annotator_json_boxes_list: Optional[List[Any]] = None,
        fuse_initial_query: bool = False,
    ):
        relevance_feedback_results = self.image_based_relevance_feedback(
            query=query,
            relevant_image_paths=relevant_image_paths,
            annotator_json_boxes_list=annotator_json_boxes_list,
            top_k_feedback=top_k
        )

        relevant_segments = relevance_feedback_results["relevant_segments"]
        irrelevant_segments = relevance_feedback_results["irrelevant_segments"]

        with torch.no_grad():
            if relevant_segments is not None and relevant_segments:
                positive_embeddings = self.wrapper.get_image_embeddings(
                    self.wrapper.process_inputs(images=relevant_segments)
                )
            else:
                positive_embeddings = None
            if irrelevant_segments is not None and irrelevant_segments:
                negative_embeddings = self.wrapper.get_image_embeddings(
                    self.wrapper.process_inputs(images=irrelevant_segments)
                )
            else:
                negative_embeddings = None
            
            processed_query = self.wrapper.process_inputs(text=query)
            with torch.no_grad():
                query_embedding = self.wrapper.get_text_embeddings(processed_query)

            rocchio_query_embedding = (self.accumulated_query_embeddings["query_embedding"] + query_embedding) / 2 if (
                fuse_initial_query
            ) else self.accumulated_query_embeddings["query_embedding"]

            self.accumulated_query_embeddings["query_embedding"] = self.rocchio_update(
                query_embeddings=rocchio_query_embedding,
                positive_embeddings=positive_embeddings,
                negative_embeddings=negative_embeddings,
            )
        
        scores, img_ids = self.index.search(
            self.accumulated_query_embeddings["query_embedding"],
            top_k
        )
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        retrieved_images = resize_images(retrieved_images, self.config)

        self.retrieval_round += 1

        return retrieved_images, scores, retrieved_image_paths