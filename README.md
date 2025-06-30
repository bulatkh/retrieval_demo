# VisualReF: Visual Relevance Feedback Prototype for Interactive Image Retrieval

## Getting started

Python version: 3.11

We use venv for managing project dependencies.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

We use two open-source datasets as use-cases for our demo:

1. General image search with COCO dataset:

    Data preparation: 
    ```
    mkdir data
    mkdir data/coco

    cd data
    wget http://images.cocodataset.org/zips/train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip
    wget http://images.cocodataset.org/zips/test2014.zip

    unzip train2014.zip -d coco/
    unzip val2014.zip -d coco/
    unzip test2014.zip -d coco/
    ```

    Build faiss index with `clip-vit-large-patch14`:
    ```
    python write_faiss_index.py \
        --data data/coco/ \
        --output faiss/retail/ \
        --batch_size 64 \
        --model_family clip \
        --model_id openai/clip-vit-large-patch14
    ```

2. Retail catalogue search with Retail-786k:
    Data preparation:
    ```
    wget https://zenodo.org/records/7970567/files/retail-786k_256.zip?download=1 -O retail-768k_256.zip

    unzip retail-786k_256.zip -d data/
    ```

    Build faiss index with `clip-vit-large-patch14`:
    ```
    python write_faiss_index.py \
        --data data/retail-786k_256/ \
        --output faiss/retail/ \
        --batch_size 64 \
        --model_family clip \
        --model_id openai/clip-vit-large-patch14
    ```

## Launch the prototype

- With image database based on COCO dataset and `clip-vit-large-patch14`:
    ```
    python -m demo.app \
        --config_path configs/demo/coco_clip_large.yaml \
        --captioning_model_config_path configs/captioning/llava_8bit.yaml 
    ```

- With image database based on Retail-786k dataset and `clip-vit-large-patch14`:
    ```
    python -m demo.app \
    --config_path configs/demo/retail_clip_large.yaml \
    --captioning_model_config_path configs/captioning/retail_llava_8bit.yaml 
    ```
