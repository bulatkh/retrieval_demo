import argparse
import os
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import torch
from gradio_image_annotation import image_annotator
from PIL import Image

import faiss
from models.attentive_summarizer import init_summarizer
from models.configs import get_model_config
from models.relevance_feedback import AFSRelevanceFeedback, RocchioUpdate
from utils.utils import get_timestamp, load_yaml, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", 
        type=str, 
        help="Path to the main configuration file"
    )
    parser.add_argument(
        "--summarizer_config_path", 
        type=str, 
        default=None,
        help="Path to the summarizer configuration file"
    )
    parser.add_argument(
        "--summarizer_checkpoint_path", 
        type=str, 
        default=None,
        help="Path to the summarizer checkpoint file"
    )
    return parser.parse_args()

args = parse_args()

CONFIG_PATH = args.config_path
SUMMARIZER_CONFIG_PATH = args.summarizer_config_path
SUMMARIZER_CHECKPOINT_PATH = args.summarizer_checkpoint_path

logs = {
    "start_timestamp": get_timestamp(),
    "config_path": CONFIG_PATH,
    "summarizer_config_path": SUMMARIZER_CONFIG_PATH,
    "summarizer_checkpoint_path": SUMMARIZER_CHECKPOINT_PATH,
    "experiments": {},
}

retrieval_round = 1
experiment_id = 0

accumulated_query_embeddings = {"query_embedding": None}

config = load_yaml(CONFIG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VLM: Retrieval Backbone
model_config = get_model_config(config["VLM_MODEL_FAMILY"], config["VLM_MODEL_NAME"])
processor = model_config["processor_class"].from_pretrained(model_config["model_id"])
model = model_config["model_class"].from_pretrained(model_config["model_id"])
model.eval()
wrapper = model_config["wrapper_class"](model=model, processor=processor)

# Read image index: candidate images
index = faiss.read_index(config["INDEX_PATH"])
with open(os.path.join(os.path.dirname(config["INDEX_PATH"]), "image_paths.txt"), "r") as f:
    candidate_image_paths = [line.strip() for line in f.readlines()]

# Initialize relevance feedback model
if SUMMARIZER_CONFIG_PATH is not None and SUMMARIZER_CHECKPOINT_PATH is not None:
    summarizer_config = load_yaml(SUMMARIZER_CONFIG_PATH)
    summarizer = init_summarizer(summarizer_config, SUMMARIZER_CHECKPOINT_PATH)
    afs_relevance_feedback = AFSRelevanceFeedback(
        model_family=config["VLM_MODEL_FAMILY"],
        vlm_wrapper=wrapper,
        summarizer=summarizer,
        img_size=model.config.vision_config.image_size,
        patch_size=model.config.vision_config.patch_size,
    )
else:
    raise ValueError("Summarizer config path and checkpoint path are required")
rocchio_update = RocchioUpdate(alpha=0.5, beta=0.25, gamma=0.25)


def resize_images_with_processor(images, processor):
    images = processor.image_processor.preprocess(
        images,
        do_normalize=False,
        do_rescale=False,
        return_tensors="np"
    )["pixel_values"]
    return [Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8)) for image in images]


def update_logs_retrieval(experiment_id, retrieval_round, user_query, top_k, retrieved_image_paths, scores):
    logs["experiments"][experiment_id].append(
        {
            "timestamp": get_timestamp(),
            "type": "retrieval",
            "round": retrieval_round,
            "user_query": user_query,
            "top_k": top_k,
            "retrieved_image_paths": retrieved_image_paths,
            "scores": scores,
        }
    )
    save_json(logs, config["RETRIEVAL_LOGS_PATH"])


def update_logs_feedback(
        experiment_id: str,
        retrieval_round: int,
        user_query: str,
        annotations: List[Dict[str, Any]],
    ):
    logs["experiments"][experiment_id].append(
        {
            "timestamp": get_timestamp(),
            "type": "feedback",
            "round": retrieval_round,
            "user_query": user_query,
            "annotations": annotations,
        }
    )
    save_json(logs, config["RETRIEVAL_LOGS_PATH"])


def image_search(query, top_k=5):
    """Retrieve images based on text query"""
    global retrieval_round
    global experiment_id
    experiment_id += 1
    logs["experiments"][experiment_id] = []

    processed_query = wrapper.process_inputs(text=query)
    with torch.no_grad():
        query_embedding = wrapper.get_text_embeddings(processed_query)

    accumulated_query_embeddings["query_embedding"] = query_embedding

    scores, img_ids = index.search(query_embedding, top_k)
    scores = scores.squeeze().tolist()
    img_ids = img_ids.squeeze().tolist()
    retrieved_image_paths = [candidate_image_paths[i] for i in img_ids]
    retrieved_images = [Image.open(path) for path in retrieved_image_paths]
    retrieved_images = resize_images_with_processor(retrieved_images, wrapper.processor)

    update_logs_retrieval(experiment_id, retrieval_round, query, top_k, retrieved_image_paths, scores)

    return retrieved_images, scores, retrieved_image_paths


def process_feedback(query, top_k, image_paths, annotator_json_boxes_list):
    """Process feedback from the annotator"""

    if SUMMARIZER_CONFIG_PATH is not None:
        relevance_feedback_results = afs_relevance_feedback(
                query=query,
                relevant_image_paths=image_paths,
                visualization=True,
                top_k_feedback=top_k,
                annotator_json_boxes_list=annotator_json_boxes_list,
            )
    
    return (
        relevance_feedback_results["positive"],
        relevance_feedback_results["negative"],
        relevance_feedback_results.get("relevant_captions", ""),
        relevance_feedback_results.get("irrelevant_captions", ""),
        relevance_feedback_results["explanation"],
    )


def feedback_loop(
    query, 
    top_k, 
    image_paths, 
    annotator_json_boxes_list,
    fuse_initial_query: bool = False
):
    """Apply feedback to the image search"""
    global retrieval_round
    
    processed_query = wrapper.process_inputs(text=query)
    with torch.no_grad():
        query_embedding = wrapper.get_text_embeddings(processed_query)

    if SUMMARIZER_CONFIG_PATH is not None:
        relevance_feedback_results = afs_relevance_feedback(
                query=query,
                relevant_image_paths=image_paths,
                visualization=True,
                top_k_feedback=top_k,
                annotator_json_boxes_list=annotator_json_boxes_list
            )
    else:
        raise ValueError("Summarizer config path and checkpoint path are required")

    rocchio_query_embedding = (accumulated_query_embeddings["query_embedding"] + query_embedding) / 2 if (
        fuse_initial_query
    ) else accumulated_query_embeddings["query_embedding"]
    accumulated_query_embeddings["query_embedding"] = rocchio_update(
        query_embeddings=rocchio_query_embedding,
        positive_embeddings=relevance_feedback_results["positive"],
        negative_embeddings=relevance_feedback_results["negative"]
    )

    scores, img_ids = index.search(accumulated_query_embeddings["query_embedding"], top_k)
    scores = scores.squeeze().tolist()
    img_ids = img_ids.squeeze().tolist()
    retrieved_image_paths = [candidate_image_paths[i] for i in img_ids]
    retrieved_images = [Image.open(path) for path in retrieved_image_paths]
    retrieved_images = resize_images_with_processor(retrieved_images, wrapper.processor)

    update_logs_feedback(
        experiment_id,
        retrieval_round,
        query,
        annotator_json_boxes_list,
    )

    retrieval_round += 1

    update_logs_retrieval(experiment_id, retrieval_round, query, top_k, retrieved_image_paths, scores)

    return (
        retrieved_images, 
        scores, 
        retrieved_image_paths, 
        relevance_feedback_results["explanation"]
    )


def get_boxes_json(annotations):
    """Get bounding boxes from annotator"""
    return annotations["boxes"] if annotations["boxes"] else None


css = """
#warning {background-color: #FFCCCB} 
.feedback {font-size: 20px !important;}
.feedback textarea {font-size: 20px !important;}
"""

with gr.Blocks(title="Multimodal Retrieval Demo", css=css) as demo:
    gr.Markdown("# Text-to-Image Search")

    image_top_k = gr.State(value=5)
    fuse_initial_query = gr.State(value=config.get("FUSE_INITIAL_QUERY", False))

    with gr.Tab("Image Search"):
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="Describe the image you would like to find:")
                image_search_btn = gr.Button("Search Images")

        annotators = []
        annotator_json_boxes_list = []

        with gr.Row():
            image_gallery = gr.Gallery(
                label="Retrieved Images", columns=5, rows=1, visible=config["SHOW_IMAGE_GALLERY"]
            )

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

        def format_outputs_image_search(images, scores, retrieved_image_paths):
            outputs_annotators = []
            outputs_gallery = []
            outputs_retrieved_image_paths = []
            outputs_images_with_saliency = None
            for i in range(len(images)):
                outputs_annotators.append({"image": images[i]})
                outputs_gallery.append((images[i], f"Relevance score: {scores[i]}"))
                outputs_retrieved_image_paths.append(retrieved_image_paths[i])
            final_outputs = [outputs_gallery] \
                + [outputs_retrieved_image_paths] \
                + [outputs_images_with_saliency] \
                + outputs_annotators
            return final_outputs
        
        relevant_image_paths = gr.State(value=None)

        with gr.Row():
            process_feedback_btn = gr.Button("Process feedback")
            
        with gr.Row():
            image_attention_gallery = gr.Gallery(
                label="Feedback Explanations (Previous Round)", columns=5, rows=1, visible=config["SHOW_IMAGE_GALLERY"]
            )

        image_search_btn.click(
            fn=lambda query, top_k: format_outputs_image_search(*image_search(query, top_k)),
            inputs=[query, image_top_k],
            outputs=[image_gallery, relevant_image_paths, image_attention_gallery, *annotators],
        )

        with gr.Row():
            with gr.Column():
                    relevant_features = gr.Textbox(
                        label="Relevant features",
                        visible=False,
                        interactive=True,
                        elem_classes=["feedback"]
                    )
            with gr.Column():
                    irrelevant_features = gr.Textbox(
                        label="Irrelevant features",
                        visible=False,
                        interactive=True,
                        elem_classes=["feedback"]
                    )


        def format_outputs_process_feedback(positive, negative, relevant_captions, irrelevant_captions, explanation):
            outputs_explanation = []
            for i in range(len(explanation)):
                outputs_explanation.append(explanation[i])
            for i, caption in enumerate(relevant_captions):
                if caption.endswith("."):
                    relevant_captions[i] = caption[:-1]
            outputs_relevant_captions = ", ".join(relevant_captions)
            for i, caption in enumerate(irrelevant_captions):
                if caption.endswith("."):
                    irrelevant_captions[i] = caption[:-1]
            outputs_irrelevant_captions = ", ".join(irrelevant_captions)
            final_outputs =  [outputs_relevant_captions] \
                + [outputs_irrelevant_captions] \
                + [outputs_explanation]
            return final_outputs
            

        process_feedback_btn.click(
            fn=lambda query, top_k, image_paths, *annotator_json_boxes_list: format_outputs_process_feedback(
                *process_feedback(query, top_k, image_paths, annotator_json_boxes_list)
            ),
            inputs=[query, image_top_k, relevant_image_paths, *annotator_json_boxes_list],
            outputs=[relevant_features, irrelevant_features, image_attention_gallery],
        )

        with gr.Row():
            apply_feedback_btn = gr.Button("Apply Feedback")

        def format_outputs_feedback(images, scores, retrieved_image_paths, images_with_saliency):
            outputs_annotators = []
            outputs_gallery = []
            outputs_retrieved_image_paths = []  
            for i in range(len(images)):
                outputs_annotators.append({"image": images[i], "boxes": []})
                outputs_gallery.append((images[i], f"Relevance score: {scores[i]}"))
                outputs_retrieved_image_paths.append(retrieved_image_paths[i])
            final_outputs = [outputs_gallery] \
                + [outputs_retrieved_image_paths] \
                + outputs_annotators
            return final_outputs


        def feedback_interface(
            query,
            top_k,
            image_paths,
            fuse_initial_query,
            *annotator_json_boxes_list,
        ):
            results = feedback_loop(
                query,
                top_k,
                image_paths,
                annotator_json_boxes_list,
                fuse_initial_query
            )
            return format_outputs_feedback(*results)
        
        apply_feedback_btn.click(
            fn=feedback_interface,
            inputs=[query, image_top_k, relevant_image_paths, fuse_initial_query, *annotator_json_boxes_list],
            outputs=[image_gallery, relevant_image_paths, *annotators],
        ).then(
            fn=lambda: [None for _ in annotator_json_boxes_list],
            inputs=None,
            outputs=[*annotator_json_boxes_list]
        )


if __name__ == "__main__":
    demo.launch()
