"""
CS-MDSS Model Architecture
==========================
ResNet50-based feature extractor with multi-head output for:
1. Classification (7-class skin lesion classification)
2. Embedding generation (128-dim for similarity search)

Implements Metric Learning with combined CrossEntropy + TripletMargin Loss
"""

from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models


class CSMDSSEncoder(nn.Module):
    """
    Multi-head encoder for medical image analysis.

    Architecture:
        - Backbone: ResNet50 (pretrained on ImageNet)
        - Classification Head: FC -> 7 classes (DermaMNIST)
        - Embedding Head: FC -> 128-dim normalized embedding
    """

    def __init__(
        self,
        num_classes: int = 7,
        embedding_dim: int = 128,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Load pretrained ResNet50 backbone
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        # Extract feature dimension from backbone
        self.feature_dim = backbone.fc.in_features  # 2048 for ResNet50

        # Remove the original classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        # Embedding head for metric learning
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning both classification logits and embeddings.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dict containing:
                - 'logits': Classification logits (B, num_classes)
                - 'embedding': L2-normalized embedding (B, embedding_dim)
                - 'features': Raw backbone features (B, feature_dim)
        """
        # Extract backbone features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # (B, 2048)

        # Classification output
        logits = self.classifier(features)

        # Embedding output (L2 normalized for cosine similarity)
        embedding = self.embedding_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)

        return {"logits": logits, "embedding": embedding, "features": features}

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the embedding for similarity search."""
        with torch.no_grad():
            output = self.forward(x)
        return output["embedding"]


class TripletMiningLoss(nn.Module):
    """
    Online triplet mining with semi-hard negative selection.
    Combines with cosine similarity constraint for better embedding space.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with online mining.

        Args:
            embeddings: (B, embedding_dim) normalized embeddings
            labels: (B,) class labels

        Returns:
            Triplet loss scalar
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        if batch_size < 3:
            return torch.tensor(0.0, device=device)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        total_loss = torch.tensor(0.0, device=device)
        valid_triplets = 0

        for i in range(batch_size):
            anchor_label = labels[i]

            # Find positive indices (same class, different sample)
            positive_mask = (labels == anchor_label) & (
                torch.arange(batch_size, device=device) != i
            )
            positive_indices = torch.where(positive_mask)[0]

            # Find negative indices (different class)
            negative_mask = labels != anchor_label
            negative_indices = torch.where(negative_mask)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            # Get anchor-positive distance (use hardest positive)
            ap_distances = distances[i, positive_indices]
            hardest_positive_idx = positive_indices[ap_distances.argmax()]
            ap_dist = distances[i, hardest_positive_idx]

            # Semi-hard negative mining: find negatives that are farther than positive
            # but still within margin
            an_distances = distances[i, negative_indices]

            # Select semi-hard negatives
            semi_hard_mask = (an_distances > ap_dist) & (
                an_distances < ap_dist + self.margin
            )

            if semi_hard_mask.sum() > 0:
                # Use semi-hard negative
                semi_hard_indices = negative_indices[semi_hard_mask]
                negative_idx = semi_hard_indices[0]
            else:
                # Fall back to hardest negative
                negative_idx = negative_indices[an_distances.argmin()]

            # Compute triplet loss for this anchor
            anchor = embeddings[i : i + 1]
            positive = embeddings[hardest_positive_idx : hardest_positive_idx + 1]
            negative = embeddings[negative_idx : negative_idx + 1]

            loss = self.triplet_loss(anchor, positive, negative)
            total_loss += loss
            valid_triplets += 1

        if valid_triplets > 0:
            return total_loss / valid_triplets
        return torch.tensor(0.0, device=device)


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity constraint loss.
    Encourages same-class samples to have high cosine similarity
    and different-class samples to have low similarity.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss.

        Args:
            embeddings: (B, embedding_dim) normalized embeddings
            labels: (B,) class labels

        Returns:
            Cosine similarity loss scalar
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Compute cosine similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()

        # Remove diagonal (self-similarity)
        eye_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - eye_mask

        # Negative mask
        negative_mask = 1.0 - positive_mask - eye_mask

        # Loss: maximize similarity for positives, minimize for negatives
        positive_loss = (1 - sim_matrix) * positive_mask
        negative_loss = F.relu(sim_matrix - self.margin) * negative_mask

        # Average over valid pairs
        num_positives = positive_mask.sum()
        num_negatives = negative_mask.sum()

        loss = torch.tensor(0.0, device=device)
        if num_positives > 0:
            loss += positive_loss.sum() / num_positives
        if num_negatives > 0:
            loss += negative_loss.sum() / num_negatives

        return loss


class CSMDSSLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training CS-MDSS encoder.

    Combines:
        - CrossEntropyLoss for classification
        - TripletMarginLoss for metric learning
        - CosineSimilarityLoss for embedding quality
    """

    def __init__(
        self,
        num_classes: int = 7,
        embedding_dim: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        ce_weight: float = 1.0,
        triplet_weight: float = 0.5,
        cosine_weight: float = 0.3,
        triplet_margin: float = 0.3,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = CSMDSSEncoder(num_classes=num_classes, embedding_dim=embedding_dim)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletMiningLoss(margin=triplet_margin)
        self.cosine_loss = CosineSimilarityLoss()

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def _compute_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss for a batch."""
        images, labels = batch
        labels = labels.view(-1).long()

        # Forward pass
        outputs = self.model(images)
        logits = outputs["logits"]
        embeddings = outputs["embedding"]

        # Classification loss
        ce_loss = self.ce_loss(logits, labels)

        # Metric learning losses
        triplet_loss = self.triplet_loss(embeddings, labels)
        cosine_loss = self.cosine_loss(embeddings, labels)

        # Combined loss
        total_loss = (
            self.hparams.ce_weight * ce_loss
            + self.hparams.triplet_weight * triplet_loss
            + self.hparams.cosine_weight * cosine_loss
        )

        # Compute accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log(
            f"{stage}_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(f"{stage}_ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_triplet_loss", triplet_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_cosine_loss", cosine_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "triplet_loss": triplet_loss,
            "cosine_loss": cosine_loss,
            "acc": acc,
        }

    def training_step(self, batch, batch_idx):
        result = self._compute_loss(batch, "train")
        self.training_step_outputs.append(result)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result = self._compute_loss(batch, "val")
        self.validation_step_outputs.append(result)
        return result["loss"]

    def test_step(self, batch, batch_idx):
        result = self._compute_loss(batch, "test")
        return result["loss"]

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class GradCAM:
    """
    Grad-CAM implementation for visual explanations.
    Generates heatmaps showing which regions influenced the prediction.
    """

    def __init__(self, model: CSMDSSEncoder, target_layer: str = "backbone.7"):
        self.model = model
        self.model.eval()

        # Get target layer
        self.target_layer = self._get_layer(target_layer)

        # Storage for gradients and activations
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _get_layer(self, layer_name: str) -> nn.Module:
        """Get layer by name (e.g., 'backbone.7' for last conv block)."""
        parts = layer_name.split(".")
        layer: nn.Module = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]  # type: ignore
            else:
                layer = getattr(layer, part)
        return layer

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class for visualization. If None, uses predicted class.

        Returns:
            Tuple of (heatmap tensor, predicted class)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        outputs = self.model(input_tensor)
        logits = outputs["logits"]

        # Get target class
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        # Check that gradients and activations were captured
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. Check hook registration."
            )

        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activations (clone to avoid modifying stored tensor)
        weighted_activations = self.activations.clone()
        for i in range(weighted_activations.shape[1]):
            weighted_activations[:, i, :, :] *= pooled_gradients[i]

        # Generate heatmap
        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        return heatmap, target_class


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> CSMDSSEncoder:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        device: Target device

    Returns:
        Loaded CSMDSSEncoder model
    """
    lightning_module = CSMDSSLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model = lightning_module.model
    model.to(device)
    model.eval()
    return model


def create_model(
    num_classes: int = 7,
    embedding_dim: int = 128,
    pretrained: bool = True,
    device: str = "cpu",
) -> CSMDSSEncoder:
    """
    Create a new model instance.

    Args:
        num_classes: Number of classification classes
        embedding_dim: Dimension of embedding vector
        pretrained: Whether to use pretrained backbone
        device: Target device

    Returns:
        CSMDSSEncoder model
    """
    model = CSMDSSEncoder(
        num_classes=num_classes, embedding_dim=embedding_dim, pretrained=pretrained
    )
    model.to(device)
    return model


# DermaMNIST class labels for clinical interpretation
DERMAMNIST_LABELS = {
    0: "Actinic Keratoses (akiec)",
    1: "Basal Cell Carcinoma (bcc)",
    2: "Benign Keratosis (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic Nevi (nv)",
    6: "Vascular Lesions (vasc)",
}

# Clinical descriptions for each class
CLINICAL_DESCRIPTIONS = {
    0: "Actinic Keratoses: Precancerous rough, scaly patches caused by sun exposure. Requires monitoring and potential treatment.",
    1: "Basal Cell Carcinoma: Most common type of skin cancer. Usually appears as a slightly transparent bump. Rarely metastasizes but should be treated.",
    2: "Benign Keratosis: Non-cancerous skin growth including seborrheic keratoses. Usually harmless but may be removed for cosmetic reasons.",
    3: "Dermatofibroma: Benign fibrous nodule, often on the legs. Usually harmless and requires no treatment.",
    4: "Melanoma: The most serious type of skin cancer. Early detection and treatment are critical for survival.",
    5: "Melanocytic Nevi: Common moles. Benign but should be monitored for changes (ABCDE criteria).",
    6: "Vascular Lesions: Including angiomas, pyogenic granulomas. Usually benign but may require treatment if symptomatic.",
}

# Risk levels for clinical decision support
RISK_LEVELS = {
    0: "MODERATE",  # Actinic Keratoses - precancerous
    1: "HIGH",  # Basal Cell Carcinoma - cancer
    2: "LOW",  # Benign Keratosis
    3: "LOW",  # Dermatofibroma
    4: "CRITICAL",  # Melanoma - most dangerous
    5: "LOW",  # Melanocytic Nevi
    6: "LOW",  # Vascular Lesions
}
