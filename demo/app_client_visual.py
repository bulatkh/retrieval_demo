import argparse
import asyncio
import logging
from typing import List, Optional

import gradio as gr
from gradio_image_annotation import image_annotator

from client.retrieval_client_visual import RemoteRetrievalClientVisual
from utils.image_utils import resize_images
from utils.utils import get_timestamp, load_yaml, save_json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval Demo Visual Client")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the main configuration file"
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the retrieval server"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Gradio interface"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the interface"
    )
    return parser.parse_args()


args = parse_args()

# Load configuration
config = load_yaml(args.config_path)
logger.info(f"Loaded config from {args.config_path}")

# Initialize retrieval client
retrieval_client = RemoteRetrievalClientVisual(server_url=args.server_url)
logger.info(f"Initialized visual client for {args.server_url}")

# Initialize logging
logs = {
    "start_timestamp": get_timestamp(),
    "config_path": args.config_path,
    "server_url": args.server_url,
    "experiments": {},
}

retrieval_round = 1
experiment_id = 0

# Functions calling the server: image search
async def image_search(search_query: str, top_k: int = 5):
    """Retrieve images based on text query"""
    global retrieval_round, experiment_id
    experiment_id += 1
    logs["experiments"][experiment_id] = []

    try:
        logger.info(f"Searching for: {search_query}")
        images, scores, retrieved_image_paths = await retrieval_client.search_images(search_query, top_k)

        update_logs_retrieval(
            experiment_id,
            retrieval_round,
            search_query,
            top_k,
            retrieved_image_paths,
            scores
        )
        logger.info(f"Search completed successfully, found {len(images)} images")

        return images, scores, retrieved_image_paths
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return [], [], []


# Functions calling the server: apply feedback immediately
async def apply_feedback_immediate(
    feedback_query: str,
    top_k: int,
    relevant_image_paths: List[str],
    annotator_boxes: List,
    fuse_query: bool = False
):
    """Apply feedback immediately based on annotation boxes"""
    global retrieval_round

    try:
        logger.info(f"Applying feedback immediately for query: {feedback_query}")

        images, scores, retrieved_image_paths = await retrieval_client.apply_feedback(
            query=feedback_query,
            top_k=top_k,
            relevant_image_paths=relevant_image_paths,
            annotator_json_boxes_list=annotator_boxes,
            fuse_initial_query=fuse_query
        )

        retrieval_round += 1
        update_logs_retrieval(
            experiment_id,
            retrieval_round,
            feedback_query,
            top_k,
            retrieved_image_paths,
            scores
        )

        logger.info(f"Feedback applied successfully, found {len(images)} images")
        return images, scores, retrieved_image_paths
    except Exception as e:
        logger.error(f"Apply feedback failed: {str(e)}")
        return [], [], []

# Update logs after retrieval
def update_logs_retrieval(
    experiment_id: int,
    retrieval_round: int,
    user_query: str,
    top_k: int,
    retrieved_image_paths: List[str],
    scores: List[float],
):
    """Update logs with retrieval information"""
    logs["experiments"][experiment_id].append({
        "timestamp": get_timestamp(),
        "type": "retrieval",
        "round": retrieval_round,
        "user_query": user_query,
        "top_k": top_k,
        "retrieved_image_paths": retrieved_image_paths,
        "scores": scores,
    })
    try:
        save_json(logs, config["RETRIEVAL_LOGS_PATH"])
    except Exception as e:
        logger.warning(f"Failed to save logs: {str(e)}")

# Update logs after feedback
def update_logs_feedback(
    exp_id: int,
    round_num: int,
    user_query: str,
    annotations: List,
):
    """Update logs with feedback information"""
    logs["experiments"][exp_id].append({
        "timestamp": get_timestamp(),
        "type": "feedback",
        "round": round_num,
        "user_query": user_query,
        "annotations": annotations,
    })
    try:
        save_json(logs, config["RETRIEVAL_LOGS_PATH"])
    except Exception as e:
        logger.warning(f"Failed to save logs: {str(e)}")


def get_boxes_json(annotations) -> Optional[List]:
    """Get bounding boxes from annotator"""
    return annotations["boxes"] if annotations["boxes"] else None


def format_outputs_image_search(images: List, scores: List[float], retrieved_image_paths: List[str]):
    """Format outputs for image search"""
    outputs_annotators = []
    outputs_gallery = []
    outputs_retrieved_image_paths = []

    images = resize_images(images, config)

    for idx in range(len(images)):
        outputs_annotators.append({"image": images[idx]})
        outputs_gallery.append((images[idx], f"Relevance score: {scores[idx]:.4f}"))
        outputs_retrieved_image_paths.append(retrieved_image_paths[idx])

    final_outputs = [outputs_gallery] + [outputs_retrieved_image_paths] + outputs_annotators
    return final_outputs


def format_outputs_feedback(
        images: List,
        scores: List[float],
        retrieved_image_paths: List[str]
):
    """Format outputs for feedback"""
    outputs_annotators = []
    outputs_gallery = []
    outputs_retrieved_image_paths = []

    images = resize_images(images, config)

    for idx in range(len(images)):
        outputs_annotators.append({"image": images[idx], "boxes": []})
        outputs_gallery.append((images[idx], f"Relevance score: {scores[idx]:.4f}"))
        outputs_retrieved_image_paths.append(retrieved_image_paths[idx])

    final_outputs = [outputs_gallery] + [outputs_retrieved_image_paths] + outputs_annotators
    return final_outputs


# Start of the Gradio interface
css = """
#warning {background-color: #FFCCCB}
.feedback {font-size: 20px !important;}
.feedback textarea {font-size: 20px !important;}
.server-status {background-color: #E8F5E8; padding: 10px; border-radius: 5px; margin: 10px 0;}
.error-message {background-color: #FFE6E6; padding: 10px; border-radius: 5px; margin: 10px 0;}
"""

with gr.Blocks(title="Multimodal Retrieval Demo - Visual Client", css=css) as demo:
    gr.Markdown("# Text-to-Image Search (Visual Client)")

    image_top_k = gr.State(value=config.get("TOP_K", 5))
    fuse_initial_query = gr.State(value=config.get("FUSE_INITIAL_QUERY", True))

    with gr.Tab("Image Search"):
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(
                    label="Describe the image you would like to find:",
                    placeholder="Enter your search query here..."
                )
                image_search_btn = gr.Button("Search Images", variant="primary")

                error_display = gr.HTML(visible=False)

        with gr.Row():
            image_gallery = gr.Gallery(
                label="Retrieved Images",
                columns=5,
                rows=1,
                visible=config["SHOW_IMAGE_GALLERY"],
                show_label=True
            )

        # Annotators for feedback
        annotators = []
        annotator_json_boxes_list = []

        with gr.Row():
            for i in range(image_top_k.value):
                with gr.Column():
                    annotator = image_annotator(
                        value=None,
                        label_list=["Relevant", "Irrelevant"],
                        label_colors=[(0, 255, 0), (255, 0, 0)],
                        label=f"Result {i + 1}",
                        visible=config["SHOW_ANNOTATORS"],
                        sources=[],
                    )
                    annotators.append(annotator)
                    button_get = gr.Button(f"Get bounding boxes for Result {i + 1}")
                    annotator_json_boxes = gr.JSON(visible=True)
                    annotator_json_boxes_list.append(annotator_json_boxes)
                    button_get.click(get_boxes_json, inputs=annotator, outputs=annotator_json_boxes)

        relevant_image_paths = gr.State(value=None)

        with gr.Row():
            apply_feedback_btn = gr.Button("Apply Feedback", variant="primary")

        async def handle_image_search(search_query, top_k):
            try:
                if not search_query.strip():
                    error_msg = '<div class="error-message">Please enter a search query.</div>'
                    return [gr.HTML(visible=True, value=error_msg)] + [None] * (len(annotators) + 2)

                # Run async search
                images, scores, retrieved_image_paths = await image_search(search_query, top_k)

                if not images:
                    error_msg = '<div class="error-message">No images found or server error occurred.</div>'
                    return [gr.HTML(visible=True, value=error_msg)] + [None] * (len(annotators) + 2)

                formatted_outputs = format_outputs_image_search(images, scores, retrieved_image_paths)
                return [gr.HTML(visible=False)] + formatted_outputs
            except Exception as e:
                logger.error(f"Image search error: {str(e)}")
                error_msg = f'<div class="error-message">Error: {str(e)}</div>'
                return [gr.HTML(visible=True, value=error_msg)] + [None] * (len(annotators) + 2)

        image_search_btn.click(
            fn=handle_image_search,
            inputs=[query, image_top_k],
            outputs=[error_display, image_gallery, relevant_image_paths, *annotators],
        )

        # Apply feedback button handler
        async def handle_apply_feedback(
            feedback_query,
            top_k,
            image_paths,
            fuse_query,
            *annotator_boxes
        ):
            try:
                if not feedback_query.strip():
                    error_msg = '<div class="error-message">Please enter a feedback query.</div>'
                    return [gr.HTML(visible=True, value=error_msg)] + [None] * (len(annotators) + 2)

                if not image_paths:
                    error_msg = '<div class="error-message">Please search for images first.</div>'
                    return [gr.HTML(visible=True, value=error_msg)] + [None] * (len(annotators) + 2)

                logger.info(f"Applying feedback: {feedback_query}, {top_k}, {image_paths}, {list(annotator_boxes)}")
                
                # Update logs with feedback information
                update_logs_feedback(
                    experiment_id,
                    retrieval_round,
                    feedback_query,
                    list(annotator_boxes)
                )

                images, scores, retrieved_image_paths = await apply_feedback_immediate(
                    feedback_query,
                    top_k,
                    image_paths,
                    list(annotator_boxes),
                    fuse_query
                )

                if not images:
                    error_msg = '<div class="error-message">No images found or server error occurred.</div>'
                    return [gr.HTML(visible=True, value=error_msg)] + [None] * (len(annotators) + 2)

                formatted_outputs = format_outputs_feedback(
                    images,
                    scores,
                    retrieved_image_paths
                )
                # formatted_outputs = [gallery, retrieved_image_paths, *annotators]
                # Expected outputs: [error_display, image_gallery, relevant_image_paths, *annotators]
                gallery, retrieved_paths, *annotator_outputs = formatted_outputs
                return [gr.HTML(visible=False), gallery, retrieved_paths] + annotator_outputs
            except Exception as e:
                logger.error(f"Apply feedback error: {str(e)}")
                error_msg = f'<div class="error-message">Error: {str(e)}</div>'
                return [gr.HTML(visible=True, value=error_msg)] + [None] * (len(annotators) + 2)

        apply_feedback_btn.click(
            fn=handle_apply_feedback,
            inputs=[query, image_top_k, relevant_image_paths, fuse_initial_query, *annotator_json_boxes_list],
            outputs=[error_display, image_gallery, relevant_image_paths, *annotators],
        ).then(
            fn=lambda: [None for _ in annotator_json_boxes_list],
            inputs=None,
            outputs=[*annotator_json_boxes_list]
        )

    # Server health check
    with gr.Tab("Server Status"):
        gr.Markdown("## Server Information")

        async def check_server_health():
            try:
                response = await retrieval_client.health()
                return response
            except Exception as e:
                logger.error(f"Server health check error: {str(e)}")
                return None

        health_check_btn = gr.Button("Check Server Health")
        health_display = gr.JSON()

        health_check_btn.click(
            fn=check_server_health,
            outputs=[health_display]
        )

# Cleanup function
async def cleanup():
    """Cleanup resources"""
    try:
        await retrieval_client.close()
        logger.info("Client cleanup completed")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")


if __name__ == "__main__":
    try:
        logger.info(f"Starting visual client on port {args.port}")
        demo.launch(
            server_port=args.port,
            share=args.share,
            show_error=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down visual client...")
    except Exception as e:
        logger.error(f"Visual client startup error: {str(e)}")
    finally:
        # Cleanup
        asyncio.run(cleanup())
