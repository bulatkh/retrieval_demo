from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from models.attentive_summarizer import AttentiveSummarizer
from models.vlm_wrapper import VLMWrapperCaptioning, VLMWrapperRetrieval


class RocchioUpdate:
    def __init__(self, alpha: float = 0.8, beta: float = 0.1, gamma: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: Optional[torch.Tensor] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        norm_output: bool = True
    ):
        return self.rocchio_update(
            query_embeddings,
            positive_embeddings,
            negative_embeddings,
            self.alpha,
            self.beta,
            self.gamma,
            norm_output
        )

    def rocchio_update(
        self,
        query_embeddings: torch.Tensor,
        avg_relevance_vector: Optional[torch.Tensor] = None,
        avg_non_relevance_vector: Optional[torch.Tensor] = None,
        alpha: float = 0.8,
        beta: float = 0.1,
        gamma: float = 0.1,
        norm_output: bool = True
    ):
        """
        Update the query embeddings using Rocchio's algorithm
            upd_q = alpha * q + beta * positive_feedback - gamma * negative_feedback

        Args:
            query_embedddings: initial query embeddings
            avg_relevance_vector: average relevance (positive feedback) vector
            avg_non_relevance_vector: average non-relevance (negative feedback) vector
            alpha: coefficient for initial query embeddings
            beta: coefficient for positive feedback
            gamma: coefficient for negative feedback
            norm_output: whether to normalize the output
        
        If both avg_relevance_vector and avg_non_relevance_vector are None or beta and gamma are 0,
        the query embeddings are returned unchanged.
        """
        if avg_non_relevance_vector is None:
            avg_non_relevance_vector = torch.zeros_like(query_embeddings)
            gamma = 0.0
        if avg_relevance_vector is None:
            avg_relevance_vector = torch.zeros_like(query_embeddings)
            beta = 0.0
        updated_query_embeddings = (
            alpha * query_embeddings + \
            beta * avg_relevance_vector - \
            gamma * avg_non_relevance_vector
        )
        if norm_output:
            updated_query_embeddings = F.normalize(updated_query_embeddings, p=2, dim=-1)
        return updated_query_embeddings


class RelevanceFeedback(ABC):
    """
    Abstract class for relevance feedback models.

    Instances are callable and require at least a query.
    """

    @abstractmethod
    def __call__(self, query: str, *args, **kwargs):
        pass


class AFSRelevanceFeedback(RelevanceFeedback):
    def __init__(
            self,
            model_family: str,
            vlm_wrapper: VLMWrapperRetrieval,
            summarizer: AttentiveSummarizer,
            temperature: float = 0.05,
            img_size: int = 224,
            patch_size: int = 32,
            user_feedback_weight: float = 5.
        ):
        self.model_family = model_family
        self.vlm_wrapper = vlm_wrapper
        self.summarizer = summarizer
        self.temperature = temperature
        self.img_size = img_size
        self.patch_size = patch_size
        self.user_feedback_weight = user_feedback_weight

    def __call__(
        self,
        query: str,
        relevant_image_paths: List[str],
        generative_captions: Optional[List[str]] = None,
        annotator_json_boxes_list: Optional[List[Dict[str, Any]]] = None,
        visualization: bool = False,
        top_k_feedback: int = 5,
    ) -> Dict[str, Any]:
        if len(relevant_image_paths) < top_k_feedback:
            raise ValueError(f"Number of images is less than {top_k_feedback}.")

        # To pass query and images to the VLM processor and model in batches:
        repeated_query = [query] * top_k_feedback

        images = []
        for image_path in relevant_image_paths:
            image = Image.open(image_path)
            images.append(image)

        patch_masks = None
        if annotator_json_boxes_list is not None:
            num_patches_axis = (self.img_size // self.patch_size)
            patch_masks = np.zeros((len(images), num_patches_axis, num_patches_axis))

            for i in range(len(images)):
                if annotator_json_boxes_list[i] is not None:
                    for annot in annotator_json_boxes_list[i]:
                        x_patch_idx_min = int(annot["xmin"] // self.patch_size)
                        x_patch_idx_max = int(annot["xmax"] // self.patch_size)
                        y_patch_idx_min = int(annot["ymin"] // self.patch_size)
                        y_patch_idx_max = int(annot["ymax"] // self.patch_size)
                        if annot["label"] == "Relevant":
                            patch_masks[i, y_patch_idx_min:y_patch_idx_max + 1, x_patch_idx_min:x_patch_idx_max + 1] = 1
                        elif annot["label"] == "Irrelevant":
                            patch_masks[i, y_patch_idx_min:y_patch_idx_max + 1, x_patch_idx_min:x_patch_idx_max + 1] = -1

            patch_masks = np.reshape(patch_masks, (len(images), -1))
            patch_masks = torch.tensor(patch_masks, dtype=torch.float32, device=self.vlm_wrapper.model.device)
            cls_column = torch.ones((len(images), 1), dtype=torch.float32, device=self.vlm_wrapper.model.device)
            patch_masks = torch.cat([cls_column, patch_masks], dim=1)
            patch_masks = patch_masks.flatten()

        inputs = self.vlm_wrapper.process_inputs(text=repeated_query, images=images)

        with torch.no_grad():
            vlm_outputs = self.vlm_wrapper.get_embeddings(
                inputs=inputs,
            )

            # This is not tested yet
            captions_embeddings = None
            if generative_captions is not None:
                captions_embeddings = self.vlm_wrapper.get_embeddings(
                    inputs=self.vlm_wrapper.process_inputs(
                        text=generative_captions,
                        images=images
                    )
                )["text_model_output"]
            
            # if patch_masks is not None:
            #     mask = (
            #         patch_masks
            #         .unsqueeze(0)
            #         .repeat(vlm_outputs["text_model_output"].shape[1] + 1, 1)
            #     )
            # else:
            #     mask = None
            summarized_vector, xattn = self.summarizer(
                q=vlm_outputs["text_model_output"][0].unsqueeze(0),
                text_inputs=captions_embeddings,
                vision_inputs=vlm_outputs["vision_model_output"].unsqueeze(0),
                mask=None,
                add_vals_to_xattn=patch_masks * self.user_feedback_weight
            )
            print(patch_masks.shape)

        assert summarized_vector.shape[1] == vlm_outputs["text_embeds"].shape[1]

        # We use the summarized vector as the positive relevance vector
        relevance_vector_pos = F.normalize(summarized_vector, p=2, dim=-1)

        # The negative relevance vector is computed from the cross-attention scores
        relevance_vector_neg, xattn_image_mean = self._get_negative_vector_from_xattn(
            vlm_outputs=vlm_outputs,
            xattn=xattn,
            top_k_feedback=top_k_feedback,
            captions_embeddings=captions_embeddings if generative_captions is not None else None,
            # relevance_attention_masks=patch_masks
        )

        if visualization:
            images_with_saliency = []
            saliency_maps = self._visualize_attention_images(xattn_image_mean, top_k_feedback)
            for i, saliency_map in enumerate(saliency_maps):
                image_with_saliency = self._draw_saliency_over_image(saliency_map, images[i])
                images_with_saliency.append(image_with_saliency)

        return {
            "positive": relevance_vector_pos,
            "negative": relevance_vector_neg,
            "explanation": images_with_saliency if visualization else images
        }

    def _get_negative_vector_from_xattn(
            self,
            vlm_outputs: Dict[str, Any],
            xattn: torch.Tensor,
            top_k_feedback: int,
            captions_embeddings: Optional[torch.Tensor] = None,
            relevance_attention_masks: Optional[torch.Tensor] = None
        ) -> torch.Tensor:

        xattn = xattn.squeeze()

        # Get cross-attention weights for image and text tokens
        # In case of local embeddings:
        #   number of text tokens is len_seq * args.top_k_feedback
        #   number of image tokens is num_patches * args.top_k_feedback
        image_tokens_start = xattn.shape[-1] - (vlm_outputs["vision_model_output"].shape[1] * top_k_feedback)

        xattn_text = xattn[..., :image_tokens_start] if captions_embeddings is not None else None
        xattn_image = xattn[..., image_tokens_start:]

        if xattn_text is not None:
            xattn_text_sum = (
                xattn_text.sum(dim=0).sum(dim=0)
                .reshape(top_k_feedback, -1)
                .sum(dim=1)
            ).unsqueeze(0)

            # Compute softmax of aggregated cross-attention weights
            # -xattn_text_sum is used to get NEGATIVE relevance
            xattn_text_softmax = F.softmax((-xattn_text_sum) / self.temperature, dim=0)
            neg_relevance_texts = torch.sum(
                captions_embeddings * xattn_text_softmax.unsqueeze(-1),
                dim=1
            ).squeeze()
        else:
            neg_relevance_texts = None

        xattn_image_mean = (
            xattn_image
            .sum(dim=0)
            .sum(dim=0)
        )

        if relevance_attention_masks is not None:
            print(xattn_image_mean.shape)
            print(relevance_attention_masks.shape)
            xattn_image_mean = xattn_image_mean + self.user_feedback_weight * relevance_attention_masks

        xattn_image_per_img = (
            xattn_image_mean
            .reshape(top_k_feedback, -1)
            .mean(dim=1)
            .unsqueeze(0)
        )

        xattn_image_softmax = F.softmax(
            (-xattn_image_per_img) / self.temperature,
            dim=1,
        )

        # For BLIP-2, we use the max similarity
        #   across the first dimension (learnable queries in Q-Former)
        #   to get the negative relevance vector
        if "blip2" in self.model_family:
            logits_per_image = torch.matmul(vlm_outputs["image_embeds"], vlm_outputs["text_embeds"][0].unsqueeze(0).T)
            logits_per_image, max_idx = logits_per_image.max(dim=1)
            img_embeddings_max = vlm_outputs["image_embeds"][
                torch.arange(vlm_outputs["image_embeds"].shape[0]), max_idx.squeeze()
            ]
            neg_relevance_images = torch.sum(img_embeddings_max * xattn_image_softmax.unsqueeze(-1), dim=1).squeeze()
        else:
            neg_relevance_images = torch.sum(
                vlm_outputs["image_embeds"] * xattn_image_softmax.unsqueeze(-1), dim=1
            ).squeeze()

        if neg_relevance_texts is not None:
            relevance_vector_neg = (neg_relevance_images + neg_relevance_texts) / 2
        else:
            relevance_vector_neg = neg_relevance_images
        relevance_vector_neg = F.normalize(relevance_vector_neg, p=2, dim=-1)

        return relevance_vector_neg, xattn_image_mean
    
    def _visualize_attention_images(
        self,
        xattn_image_mean: torch.Tensor,
        top_k_feedback: int,
    ):  
        attn_per_patch = (
            xattn_image_mean
            .reshape(top_k_feedback, -1)
        ).numpy()
        attn_cls = attn_per_patch[:, 0]
        attn_per_patch = attn_per_patch[:, 1:]
        saliency_maps = []

        min_attn = attn_per_patch.min()
        max_attn = attn_per_patch.max()
        for i in range(top_k_feedback):
            patch_weights = (attn_per_patch[i] - min_attn) / \
                           (max_attn - min_attn)
            grid_size = int(self.img_size // self.patch_size)
            patch_weights_grid = patch_weights.reshape(grid_size, grid_size)
            patch_weights_upsampled = np.repeat(
                np.repeat(patch_weights_grid, self.patch_size, axis=0),
                self.patch_size,
                axis=1
            )
            saliency_maps.append(patch_weights_upsampled)
        return saliency_maps

    def _draw_saliency_over_image(
        self,
        saliency_map: np.ndarray,
        image: Image.Image,
    ):  
        image = image.convert('RGB')
        saliency_map_resized = np.array(Image.fromarray(
            (saliency_map * 255).astype(np.uint8)
        ).resize(image.size, Image.BICUBIC))
        
        heatmap = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
        heatmap[:, :, 1] = saliency_map_resized  # Green channel
        
        img_array = np.array(image)
        
        alpha = 0.99
        overlay = img_array.copy()
        mask = saliency_map_resized > 0.3
        
        overlay[mask, 1] = np.clip(
            overlay[mask, 1] + (heatmap[mask, 1] * alpha),
            0, 255
        ).astype(np.uint8)
        
        overlay_image = Image.fromarray(overlay)
        return overlay_image
