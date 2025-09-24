from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from models.vlm_wrapper import VLMWrapperCaptioning, VLMWrapperRetrieval


class RocchioUpdate:
    def __init__(self, alpha: float = 0.8, beta: float = 0.1, gamma: float = 0.1, multiple: bool = False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.multiple = multiple

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: Optional[torch.Tensor] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        norm_output: bool = True
    ):
        if not self.multiple:
            return self.rocchio_update(
                query_embeddings,
                positive_embeddings,
                negative_embeddings,
                self.alpha,
                self.beta,
                self.gamma,
                norm_output
            )
        else:
            return self.rocchio_update_multiple(
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

    def rocchio_update_multiple(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: Optional[torch.Tensor] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        alpha: float = 0.8,
        beta: float = 0.1,
        gamma: float = 0.1,
        norm_output: bool = True
    ):
        """
        Update the query embeddings using Rocchio's algorithm
            upd_q = alpha * q + beta * positive_feedback - gamma * negative_feedback
        """
        if negative_embeddings is None:
            negative_embeddings = torch.zeros_like(query_embeddings)
            gamma = 0.0
        if positive_embeddings is None:
            positive_embeddings = torch.zeros_like(query_embeddings)
            beta = 0.0

        beta_feedback = beta / len(positive_embeddings)
        gamma_feedback = gamma / len(negative_embeddings)
        updated_query_embeddings = alpha * query_embeddings

        for positive_embedding in positive_embeddings:
            updated_query_embeddings += beta_feedback * positive_embedding
        for negative_embedding in negative_embeddings:
            updated_query_embeddings -= gamma_feedback * negative_embedding
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


class CaptionVLMRelevanceFeedback(RelevanceFeedback):
    def __init__(
        self,
        vlm_wrapper_retrieval: VLMWrapperRetrieval,
        vlm_wrapper_captioning: VLMWrapperCaptioning,
        img_size: int = 224,
    ):
        self.vlm_wrapper_retrieval = vlm_wrapper_retrieval
        self.vlm_wrapper_captioning = vlm_wrapper_captioning
        self.img_size = img_size

    def __call__(
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
        if len(relevant_image_paths) < top_k_feedback:
            raise ValueError(f"Number of images is less than {top_k_feedback}.")

        user_prompt = self._get_prompt(prompt_based_on_query, prompt, user_prompt)

        images = []
        image_sizes = []
        for image_path in relevant_image_paths:
            image = Image.open(image_path)
            images.append(image)
            image_sizes.append(image.size)

        images_vlm = []
        prompts_vlm = []
        relevant_mask = []
        for i in range(len(annotator_json_boxes_list)):
            if annotator_json_boxes_list[i] is not None:
                for annot in annotator_json_boxes_list[i]:
                    img = np.array(images[i].resize((self.img_size, self.img_size), Image.BICUBIC))
                    img_fragment = img[annot["ymin"]:annot["ymax"], annot["xmin"]:annot["xmax"]]
                    img_fragment = Image.fromarray(img_fragment)
                    images_vlm.append(img_fragment)
                    prompts_vlm.append(user_prompt.format(query.lower(), annot["label"].lower()))
                    relevant_mask.append(annot["label"] == "Relevant")

        if relevant_captions is None and irrelevant_captions is None:
            vlm_outputs = self._generate_captions(
                prompts_vlm=prompts_vlm,
                images_vlm=images_vlm
            )

            relevant_mask = np.array(relevant_mask)
            vlm_outputs = np.array(vlm_outputs)

            relevant_captions = vlm_outputs[relevant_mask == 1].tolist()
            irrelevant_captions = vlm_outputs[relevant_mask == 0].tolist()

        if type(relevant_captions) is str:
            relevant_captions = relevant_captions.split(", ")
        if type(irrelevant_captions) is str:
            irrelevant_captions = irrelevant_captions.split(", ")

        print("relevant_captions: ", relevant_captions)
        print("irrelevant_captions: ", irrelevant_captions)

        positive_embeddings = None
        negative_embeddings = None
        if relevant_captions:
            positive_inputs = self.vlm_wrapper_retrieval.process_inputs(
                text=relevant_captions,
            )
            with torch.no_grad():
                positive_embeddings = self.vlm_wrapper_retrieval.get_text_embeddings(
                    inputs=positive_inputs
                ).mean(dim=0)
        if irrelevant_captions:
            negative_inputs = self.vlm_wrapper_retrieval.process_inputs(
                text=irrelevant_captions,
            )
            with torch.no_grad():
                negative_embeddings = self.vlm_wrapper_retrieval.get_text_embeddings(
                    inputs=negative_inputs
                ).mean(dim=0)

        if visualization:
            images_with_captions = self._visualize_captions_on_images(
                images=images,
                annotator_json_boxes_list=annotator_json_boxes_list,
                vlm_outputs=vlm_outputs
            )

        return {
            "positive": positive_embeddings,
            "negative": negative_embeddings,
            "explanation": images_with_captions if visualization else images,
            "relevant_captions": relevant_captions,
            "irrelevant_captions": irrelevant_captions
        }

    def _get_prompt(
        self,
        prompt_based_on_query: bool,
        prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> str:
        if prompt_based_on_query:
            full_prompt = (
                "User is looking for: {}. "
                "The image is a fragment of a larger image annotated by user as {}. "
                "Describe the visual content of the image fragment in fewer than 5 words. "
            )
        else:
            full_prompt = (
                "Describe the visual content of the image fragment in fewer than 5 words. "
            )
        if user_prompt is not None:
            full_prompt = f"{full_prompt}. Focus on the following instructions: {user_prompt}"
        return full_prompt

    def _generate_captions(
        self,
        prompts_vlm: List[str],
        images_vlm: List[Image.Image]
    ) -> List[str]:
        vlm_outputs = []
        for i in range(len(prompts_vlm)):
            with torch.no_grad():
                inputs = self.vlm_wrapper_captioning.process_inputs(
                    apply_template=True,
                    image=[images_vlm[i]],
                    prompt=[prompts_vlm[i]]
                )
                vlm_output = self.vlm_wrapper_captioning.generate(inputs=inputs)
                vlm_output = self.vlm_wrapper_captioning.decode(vlm_output)
                generated_text = [text.split("ASSISTANT: ")[-1] for text in vlm_output]
                vlm_outputs.extend(generated_text)
        return vlm_outputs

    def _visualize_captions_on_images(
        self,
        images: List[Image.Image],
        annotator_json_boxes_list: List[Dict[str, Any]],
        vlm_outputs: List[str],
    ) -> List[Image.Image]:
        """Create images with caption overlays using torchvision draw_bounding_boxes"""

        images_with_captions = []
        caption_idx = 0

        for image, annotations in zip(images, annotator_json_boxes_list):
            if annotations is None:
                images_with_captions.append(image)
                continue

            # Resize image and convert to RGB if needed
            image_resized = image.resize((self.img_size, self.img_size))
            if image_resized.mode != 'RGB':
                image_resized = image_resized.convert('RGB')

            # Create a copy to draw on
            image_with_boxes = image_resized.copy()
            draw = ImageDraw.Draw(image_with_boxes)

            for annot in annotations:
                x1, y1 = annot["xmin"], annot["ymin"]
                x2, y2 = annot["xmax"], annot["ymax"]

                caption = vlm_outputs[caption_idx]
                label = f"{caption}"

                box_color = "green" if annot["label"] == "Relevant" else "red"
                text_color = "white"

                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

                try:
                    bbox = draw.textbbox((0, 0), label, font_size=20)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    text_width, text_height = draw.textsize(label)

                bg_x1 = x1
                bg_y1 = max(0, y1 - text_height - 4)
                bg_x2 = min(self.img_size, x1 + text_width + 4)
                bg_y2 = y1

                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=box_color)

                text_x = x1 + 2
                text_y = max(2, y1 - text_height - 2)
                draw.text((text_x, text_y), label, fill=text_color)

                caption_idx += 1

            images_with_captions.append(image_with_boxes)

        return images_with_captions


class ImageBasedVLMRelevanceFeedback(RelevanceFeedback):
    def __init__(
        self,
        vlm_wrapper_retrieval: VLMWrapperRetrieval,
        img_size: int = 224,
    ):
        self.vlm_wrapper_retrieval = vlm_wrapper_retrieval
        self.img_size = img_size

    def __call__(
        self,
        query: str,
        relevant_image_paths: List[str],
        annotator_json_boxes_list: Optional[List[Any]] = None,
        top_k_feedback: int = 5,
    ):
        if len(relevant_image_paths) < top_k_feedback:
            raise ValueError(f"Number of images is less than {top_k_feedback}.")

        images = []
        for image_path in relevant_image_paths:
            image = Image.open(image_path)
            images.append(image)

        segments = self._extract_image_segments(
            images=images,
            annotator_json_boxes_list=annotator_json_boxes_list
        )

        return segments

    def _extract_image_segments(
        self,
        images: List[Image.Image],
        annotator_json_boxes_list: List[Dict[str, Any]]
    ) -> List[Image.Image]:
        irrelevant_segments = []
        relevant_segments = []
        for i in range(len(annotator_json_boxes_list)):
            if annotator_json_boxes_list[i] is not None:
                for annot in annotator_json_boxes_list[i]:
                    segment = np.array(
                        images[i].resize(
                            (self.img_size, self.img_size), Image.BICUBIC
                        )
                    )[annot["ymin"]:annot["ymax"], annot["xmin"]:annot["xmax"]]
                    segment = Image.fromarray(segment).resize((self.img_size, self.img_size))
                    if annot["label"] == "Relevant":
                        relevant_segments.append(segment)
                    elif annot["label"] == "Irrelevant":
                        irrelevant_segments.append(segment)
                    else:
                        raise ValueError(f"Invalid label: {annot['label']}")
        return {
            "relevant_segments": relevant_segments,
            "irrelevant_segments": irrelevant_segments
        }
