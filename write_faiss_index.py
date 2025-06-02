import argparse
import glob
import os
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.configs import get_model_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset with images')
    parser.add_argument('--output', type=str, required=True, help='Path to output FAISS index')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_family', type=str, required=True, help='VLM model family')
    parser.add_argument('--model_id', type=str, required=True, help='HF model id')
    parser.add_argument('--index_type', type=str, default='flat_ip', help='Index type')
    parser.add_argument(
        '--m',
        type=int,
        default=32,
        help='Number of connections per layer only for `hnsw` index type.'
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device cuda/cpu for generating embeddings')
    return parser.parse_args()


def get_image_paths(data_dir):
    extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, f'**/*.{ext}'), recursive=True))
    return image_paths


def encode_images(vlm_wrapper, image_paths, batch_size):
    features = []
    paths = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_indices = []
        
        for j, path in tqdm(enumerate(batch_paths), desc="Encoding images"):
            try:
                img = Image.open(path).convert('RGB')
                # Use the processor directly on the image
                batch_images.append(img)
                valid_indices.append(j)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        processed_images = vlm_wrapper.process_inputs(images=batch_images)
        with torch.no_grad():
            outputs = vlm_wrapper.get_image_embeddings(processed_images)
            features.append(outputs.cpu().numpy())
        paths.extend(batch_paths)
    
    return np.vstack(features), paths


def create_faiss_index(features, feature_dim, index_type='flat_ip', m=32):
    faiss.normalize_L2(features)
    
    if index_type == 'flat_ip':
        index = faiss.IndexFlatIP(feature_dim)
    elif index_type == 'hnsw':
        index = faiss.IndexHNSWFlat(feature_dim, m)
    else:
        raise ValueError(f"Invalid index type: {index_type}")
    
    index.add(features)
    
    return index


def main():
    args = parse_args()
    
    # Get all image paths
    image_paths = get_image_paths(args.data)
    print(f"Found {len(image_paths)} images")

    model_config = get_model_config(args.model_family, args.model_id)

    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])
    model = model_config["model_class"].from_pretrained(model_config["model_id"])
    wrapper = model_config["wrapper_class"](model=model, processor=processor)

    model.to(args.device)
    model.eval()

    features, paths = encode_images(wrapper, image_paths, args.batch_size)
    feature_dim = features.shape[1]

    # Create FAISS index
    index = create_faiss_index(features, feature_dim)
    
    # Save index and paths
    output_dir = os.path.join(args.output, args.model_id)
    os.makedirs(output_dir, exist_ok=True)
    
    faiss.write_index(index, os.path.join(output_dir, "image_index.faiss"))
    
    # Save paths to a text file
    with open(os.path.join(output_dir, "image_paths.txt"), "w") as f:
        for path in paths:
            f.write(f"{path}\n")
    
    print(f"Index created with {len(paths)} images and saved to {output_dir}")

if __name__ == "__main__":
    main()


