import random
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models.attentive_pooler import AttentivePooler
from models.vlm_wrapper import VLMWrapperRetrieval


def init_summarizer(config, summarizer_checkpoint=None):
    summarizer = AttentiveSummarizer(
        pooler_config=config["pooler_config"],
        text_dim_local=config.get("text_dim_local", config.get("text_dim", 768)),
        text_dim_global=config.get("text_dim_global", config.get("text_dim", 768)),
        vision_dim=config["vision_dim"],
        vlm_wrapper=None,
        global_embeddings_vision=config.get("global_embeddings_vision", True),
        global_embeddings_text=config.get("global_embeddings_text", True),
        checkpoint_path=summarizer_checkpoint
    )
    summarizer.eval()
    return summarizer


class AttentiveSummarizer(nn.Module):
    def __init__(
            self,
            pooler_config: Dict[str, Any],
            text_dim_local: int,
            text_dim_global: int,
            vision_dim: int,
            vlm_wrapper: Optional[VLMWrapperRetrieval] = None,
            global_embeddings_vision: bool = False,
            global_embeddings_text: bool = False,
            checkpoint_path: Optional[str] = None,
            random_mask: bool = False
    ):
        super().__init__()
        self.config = pooler_config

        self.vlm_wrapper = vlm_wrapper
        if self.vlm_wrapper is not None:
            for param in self.vlm_wrapper.model.parameters():
                param.requires_grad = False
            self.vlm_wrapper_model = vlm_wrapper.model

        self.text_projection = nn.Linear(
            text_dim_global if global_embeddings_text else text_dim_local,
            pooler_config.get("embed_dim", 768)
        )
        self.vision_projection = nn.Linear(
            vision_dim,
            pooler_config.get("embed_dim", 768)
        )
        self.pooler = AttentivePooler(
            embed_dim=pooler_config.get("embed_dim", 768),
            num_heads=pooler_config.get("num_heads", 12),
            mlp_ratio=pooler_config.get("mlp_ratio", 4.0),
            depth=pooler_config.get("depth", 1),
            init_std=pooler_config.get("init_std", 0.02),
            qkv_bias=pooler_config.get("qkv_bias", True),
            complete_block=pooler_config.get("complete_block", True)
        )
        self.projection = nn.Linear(
            pooler_config.get("embed_dim", 768),
            text_dim_global
        )

        self.global_embeddings_vision = global_embeddings_vision
        self.global_embeddings_text = global_embeddings_text

        # Filter out the keys of the state dict that are not relevant
        if checkpoint_path is not None:
            device = next(self.parameters()).device
            checkpoint = torch.load(checkpoint_path, map_location=device)
            checkpoint_state_dict = checkpoint["state_dict"]
            checkpoint_state_dict = {k.replace("summarizer.", ""): v for k, v in checkpoint_state_dict.items()}
            if self.vlm_wrapper is None:
                checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if "vlm_wrapper" not in k}
            self.load_state_dict(checkpoint_state_dict)
            assert torch.allclose(
                checkpoint["state_dict"]["summarizer.text_projection.weight"].to(device),
                self.text_projection.weight.to(device)
            )
            assert torch.allclose(
                checkpoint["state_dict"]["summarizer.vision_projection.weight"].to(device),
                self.vision_projection.weight.to(device)
            )
            print(f"Weights successfully loaded from checkpoint: {checkpoint_path}")

        self.random_mask = random_mask

    def _get_text_features(self, inputs: Dict[str, Any], global_embeddings: bool = False):
        assert self.vlm_wrapper is not None
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

        inputs.update({
            "pixel_values": torch.randn(len(inputs["input_ids"]), 3, 224, 224).to(self.vlm_wrapper.model.device),
        })

        outputs = self.vlm_wrapper.get_embeddings(
            inputs=inputs
        )

        return outputs["text_embeds"] if global_embeddings else outputs["text_model_output"]

    def _get_vision_features(self, inputs: Dict[str, Any], global_embeddings: bool = False):
        assert self.vlm_wrapper is not None
        assert "pixel_values" in inputs

        input_ids_dummy = torch.randint(0, 100, (len(inputs["pixel_values"]), 10)).to(self.vlm_wrapper.model.device)
        inputs.update({
            "input_ids": input_ids_dummy,
            "attention_mask": torch.ones_like(input_ids_dummy).to(self.vlm_wrapper.model.device),
        })

        outputs = self.vlm_wrapper.get_embeddings(
            inputs=inputs
        )

        return outputs["image_embeds"] if global_embeddings else outputs["vision_model_output"]

    def forward(
            self,
            q: Union[torch.Tensor, Dict[str, Any]],
            text_inputs: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
            vision_inputs: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
            mask: Optional[torch.Tensor] = None
    ):
        assert text_inputs is not None or vision_inputs is not None

        if self.vlm_wrapper is not None:
            q_features = self._get_text_features(q, global_embeddings=self.global_embeddings_text)
            if text_inputs is not None:
                text_features = self._get_text_features(text_inputs, global_embeddings=self.global_embeddings_text)
            if vision_inputs is not None:
                vision_features = self._get_vision_features(vision_inputs, global_embeddings=self.global_embeddings_vision)
        else:
            q_features = q
            if text_inputs is not None:
                text_features = text_inputs
            if vision_inputs is not None:
                vision_features = vision_inputs

        q_features = self.text_projection(q_features)
        if text_inputs is not None:
            text_features = self.text_projection(text_features)
        if vision_inputs is not None:
            vision_features = self.vision_projection(vision_features)

        if q_features.ndim == 2:
            q_features = q_features.unsqueeze(1)
        if text_inputs is not None and text_features.ndim == 2:
            text_features = text_features.unsqueeze(1)
        if vision_inputs is not None and vision_features.ndim == 2:
            vision_features = vision_features.unsqueeze(1)

        if self.random_mask and self.training:
            if text_inputs is not None:
                topk = int(text_features.shape[0] // q_features.shape[0])
                num_k_mask_text = random.randint(0, topk - 1)
                top_k_mask_indices_text = torch.randperm(topk)[:num_k_mask_text]
                mask_text = torch.ones(q_features.shape[1] + 1, topk, text_features.shape[1])
                mask_text[:, top_k_mask_indices_text, :] = 0
                mask_text = mask_text.view(q_features.shape[1] + 1, -1)
            if vision_inputs is not None:
                topk = int(vision_features.shape[0] // q_features.shape[0])
                num_k_mask_vision = random.randint(0, topk - 1)
                top_k_mask_indices_vision = torch.randperm(topk)[:num_k_mask_vision]
                mask_vision = torch.ones(q_features.shape[1] + 1, topk, vision_features.shape[1])
                mask_vision[:, top_k_mask_indices_vision, :] = 0
                mask_vision = mask_vision.view(q_features.shape[1] + 1, -1)

            if text_inputs is not None and vision_features is not None:
                mask = torch.cat([mask_text, mask_vision], dim=1).to(self.vlm_wrapper.model.device)
            elif text_inputs is not None:
                mask = mask_text
            elif vision_inputs is not None:
                mask = mask_vision

        if text_inputs is not None:
            text_features = text_features.view(q_features.shape[0], -1, text_features.shape[-1])
        if vision_inputs is not None:
            vision_features = vision_features.view(q_features.shape[0], -1, vision_features.shape[-1])

        if text_inputs is not None and vision_inputs is not None:
            text_image_features = torch.cat([text_features, vision_features], dim=1)
        elif text_inputs is not None:
            text_image_features = text_features
        elif vision_inputs is not None:
            text_image_features = vision_features

        x, xattn = self.pooler(q_features, text_image_features, mask=mask)
        x = self.projection(x[:, 0, :])
        return x, xattn


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_features, gt_features):
        query_features = F.normalize(query_features, p=2, dim=1)
        gt_features = F.normalize(gt_features, p=2, dim=1)
        gt_features = (
            gt_features
            .view(query_features.shape[0], -1, gt_features.shape[-1])
            .mean(dim=1)
            .squeeze(1)
        )
        # query_features = query_features.repeat_interleave(ratio, dim=0)

        similarity_matrix = torch.mm(query_features, gt_features.t())

        diagonal_similarities = similarity_matrix.diag()

        # Compute hits@1 within a batch:
        # for each query, check if the matching gt_features (diagonal) has the highest similarity
        # by comparing it to the max non-diagonal similarity for that query
        with torch.no_grad():
            non_diagonal_similarities = similarity_matrix - torch.diag(diagonal_similarities)
            batch_hits_1 = (
                (diagonal_similarities - non_diagonal_similarities.max(dim=0)[0]) > 0
            ).sum() / query_features.shape[0]

        # Return the mean of 1 - diagonal similarities as the loss
        return {
            "loss": (1 - diagonal_similarities).mean(),
            "batch_hits@1": batch_hits_1
        }


class AlignmentAttentiveSummarizer(LightningModule):
    def __init__(
            self,
            summarizer: AttentiveSummarizer,
            pooler_config: Dict[str, Any],
            temperature: float = None,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            max_epochs: int = 100,
            random_mask: bool = False,
            no_image_loss: bool = False,
            no_caption_loss: bool = False
    ):
        super().__init__()
        self.summarizer = summarizer
        self.criterion = CosineSimilarityLoss()
        self.save_hyperparameters(ignore=["summarizer"])

    def forward(self, q, gt, text_inputs=None, vision_inputs=None):
        # Shapes:
        # q: (bsz, seq, text_dim) -- tokens or embeddings
        # text_inputs: (bsz, seq, text_dim) -- tokens or embeddings
        # vision_inputs: (bsz, seq, vision_dim) -- images or embeddings
        # gt: (bsz, text_dim) -- tokens or embeddings
        q, xattn = self.summarizer(q, text_inputs, vision_inputs)

        if self.summarizer.vlm_wrapper is not None:
            # the objective is to map summarized queries to the text embeddings
            # in this feature space produced by CLIP -> global_embeddings=True
            gt = self.summarizer._get_text_features(gt, global_embeddings=True)
        return q, gt, xattn

    def _shared_step(self, batch, batch_idx, split="train"):
        q = {
            "input_ids": batch.get("query_input_ids"),
            "attention_mask": batch.get("query_attention_mask"),
        }
        gt = {
            "input_ids": batch.get("ground_truth_input_ids"),
            "attention_mask": batch.get("ground_truth_attention_mask"),
        }

        text_inputs = None
        vision_inputs = None

        if batch.get("generated_text_input_ids") is not None:
            text_inputs = {
                "input_ids": batch.get("generated_text_input_ids"),
                "attention_mask": batch.get("generated_text_attention_mask"),
            }
        elif batch.get("text_feedback_input_ids") is not None:
            text_inputs = {
                "input_ids": batch.get("text_feedback_input_ids"),
                "attention_mask": batch.get("text_feedback_attention_mask"),
            }

        if batch.get("retrieval_results_images") is not None:
            vision_inputs = {
                "pixel_values": batch.get("retrieval_results_images"),
            }
        else:
            vision_inputs = None

        gt_image = {
            "pixel_values": batch.get("image"),
        }
        gt_image_features = self.summarizer._get_vision_features(
            inputs=gt_image,
            global_embeddings=True
        )
        q_global = self.summarizer._get_text_features(
            inputs=q,
            global_embeddings=True
        )
        q, gt, _ = self(q, gt, text_inputs, vision_inputs)
        loss = 0
        outputs = {}

        # Alignment loss with caption features
        if not self.hparams.no_caption_loss:
            caption_loss_dict = self.criterion(q, gt)
            loss += caption_loss_dict["loss"] * 0.5
            outputs[f"{split}/caption_loss"] = caption_loss_dict["loss"]
            outputs[f"{split}/caption_batch_hits@1"] = caption_loss_dict["batch_hits@1"]

        # For BLIP2, the image features are 3D with shape (bsz, num_q, text_dim)
        if gt_image_features.ndim == 3 and gt_image_features.shape[1] != 1:
            logits_per_image = torch.matmul(gt_image_features, q_global.t())
            logits_per_image, max_idx = logits_per_image.max(dim=1)
            max_idx = max_idx.diag()
            gt_image_features = gt_image_features[torch.arange(gt_image_features.shape[0]), max_idx, :]

        # Alignment loss with image features
        if not self.hparams.no_image_loss:
            image_loss_dict = self.criterion(q, gt_image_features)
            loss += image_loss_dict["loss"] * 0.5
            outputs[f"{split}/image_loss"] = image_loss_dict["loss"]
            outputs[f"{split}/image_batch_hits@1"] = image_loss_dict["batch_hits@1"]

        # Loss and metrics for the original VLM model
        with torch.no_grad():
            baseline_loss_dict = self.criterion(q_global, gt_image_features)
            outputs[f"{split}/baseline_batch_hits@1"] = baseline_loss_dict["batch_hits@1"]

        outputs[f"{split}/loss"] = loss

        return outputs

    def training_step(self, batch, batch_idx):
        step_outputs = self._shared_step(batch, batch_idx, split="train")
        self.log_dict(step_outputs, on_step=True)
        return step_outputs["train/loss"]

    def validation_step(self, batch, batch_idx):
        step_outputs = self._shared_step(batch, batch_idx, split="val")
        self.log_dict(step_outputs, on_epoch=True)

    def test_step(self, batch, batch_idx):
        step_outputs = self._shared_step(batch, batch_idx, split="test")
        self.log_dict(step_outputs, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
