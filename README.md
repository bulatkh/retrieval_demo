# Attentive Feedback Summarizer with Relevance Feedback from User Interactions

## Getting started

Python version: 3.11

We use venv for managing project dependencies.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

The demo is based on the COCO dataset:

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

Build a faiss index with `clip-vit-large-patch14` for the test set:
```
python write_faiss_index.py \
    --data data/coco/test2014 \
    --output faiss/coco/ \
    --batch_size 64 \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14
```

It is also possible to index the whole database (will take longer) with `--data data/coco/`.

## Launch the prototype

We use `clip-vit-large-patch14` with summarizer train on the COCO dataset. The weights can be downloaded from [here](). Unzip the file and make sure it is available as `checkpoints/clip-vit-large-patch14-2025-03-24_15_09_55_874696/epoch=19-val_loss=0.08.ckpt`.

Launch the demo:
    ```
    python -m demo.app \
        --config_path configs/demo/coco_clip_large.yaml \
        --summarizer_config_path configs/coco_summarizer/clip_large_local_summarizer_nocaploss.yaml \
        --summarizer_checkpoint_path checkpoints/clip-vit-large-patch14-2025-03-24_15_09_55_874696/epoch=19-val_loss=0.08.ckpt
    ```

