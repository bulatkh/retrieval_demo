from typing import Any, Dict, List, Optional, Union

import httpx

from utils.image_utils import base64_to_image


class RemoteRetrievalClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def search_images(self, query: str, top_k: int = 5):
        try:
            response = await self.client.post(
                f"{self.server_url}/search",
                json={"query": query, "top_k": top_k}
            )
            response.raise_for_status()
            data = response.json()

            # Load images from paths
            images = [base64_to_image(image) for image in data["images"]]

            return images, data["scores"], data["image_paths"]
        except Exception as e:
            raise Exception(f"Remote search failed: {str(e)}")

    async def process_feedback(
        self,
        query: str,
        relevant_image_paths: List[str],
        annotator_json_boxes_list: List[Optional[List[Dict[str, Any]]]],
        visualization: bool = False,
        top_k_feedback: int = 5,
        prompt_based_on_query: bool = False,
        relevant_captions: Optional[Union[List[str], str]] = None,
        irrelevant_captions: Optional[Union[List[str], str]] = None,
        prompt: Optional[str] = None
    ):
        try:
            response = await self.client.post(
                f"{self.server_url}/process_feedback",
                json={
                    "query": query,
                    "relevant_image_paths": relevant_image_paths,
                    "annotator_json_boxes_list": annotator_json_boxes_list,
                    "visualization": visualization,
                    "top_k_feedback": top_k_feedback,
                    "prompt_based_on_query": prompt_based_on_query,
                    "relevant_captions": relevant_captions,
                    "irrelevant_captions": irrelevant_captions,
                    "prompt": prompt
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["relevance_feedback_results"]
        except Exception as e:
            raise Exception(f"Remote process feedback failed: {str(e)}")

    async def apply_feedback(
        self,
        query: str,
        top_k: int,
        relevant_captions: Optional[List[str]] = None,
        irrelevant_captions: Optional[List[str]] = None,
        fuse_initial_query: bool = False
    ):
        try:
            relevant_captions = relevant_captions.split(",")
            irrelevant_captions = irrelevant_captions.split(",")
            response = await self.client.post(
                f"{self.server_url}/apply_feedback",
                json={
                    "query": query,
                    "top_k": top_k,
                    "relevant_captions": relevant_captions,
                    "irrelevant_captions": irrelevant_captions,
                    "fuse_initial_query": fuse_initial_query
                }
            )
            response.raise_for_status()
            data = response.json()

            images = [base64_to_image(image) for image in data["images"]]

            return images, data["scores"], data["image_paths"]
        except Exception as e:
            raise Exception(f"Remote feedback failed: {str(e)}")

    async def health(self):
        try:
            response = await self.client.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Remote health check failed: {str(e)}")

    async def close(self):
        await self.client.aclose()
