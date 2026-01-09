"""
CS-MDSS Utility Functions
=========================
Contains:
- Data loading and preprocessing for DermaMNIST
- Faiss index building and similarity search
- Counterfactual analysis (mock causal inference)
- Helper functions for visualization
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from medmnist import DermaMNIST
except ImportError:
    DermaMNIST = None


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================


def get_transforms(image_size: int = 224, training: bool = False) -> transforms.Compose:
    """
    Get image transforms for DermaMNIST.

    Args:
        image_size: Target image size
        training: Whether to apply training augmentations

    Returns:
        Composed transforms
    """
    if training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def inverse_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Inverse normalization for visualization.

    Args:
        tensor: Normalized image tensor

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)

    return tensor * std + mean


def load_dermamnist_datasets(
    data_dir: str = "./data", image_size: int = 224, download: bool = True
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load DermaMNIST datasets.

    Args:
        data_dir: Directory to store data
        image_size: Image size (28 or 224)
        download: Whether to download if not present

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if DermaMNIST is None:
        raise ImportError("medmnist package not installed. Run: pip install medmnist")

    train_transform = get_transforms(image_size=image_size, training=True)
    test_transform = get_transforms(image_size=image_size, training=False)

    train_dataset = DermaMNIST(
        split="train",
        transform=train_transform,
        download=download,
        root=data_dir,
        size=image_size,
    )

    val_dataset = DermaMNIST(
        split="val",
        transform=test_transform,
        download=download,
        root=data_dir,
        size=image_size,
    )

    test_dataset = DermaMNIST(
        split="test",
        transform=test_transform,
        download=download,
        root=data_dir,
        size=image_size,
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for triplet loss
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Faiss Similarity Search Engine
# ============================================================================


class SimilarityEngine:
    """
    Faiss-based similarity search engine for medical image retrieval.
    Implements the "Face Recognition" logic for finding similar cases.
    """

    def __init__(self, embedding_dim: int = 128, use_gpu: bool = False):
        """
        Initialize the similarity engine.

        Args:
            embedding_dim: Dimension of embeddings
            use_gpu: Whether to use GPU for search (requires faiss-gpu)
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu

        # Initialize Faiss index (L2 distance for normalized embeddings)
        # Using IndexFlatIP for cosine similarity (since embeddings are L2-normalized)
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Storage for metadata
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.images: Optional[np.ndarray] = None
        self.sample_indices: Optional[np.ndarray] = None

        self.is_built = False

    def build_index(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        device: str = "cpu",
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Build Faiss index from dataset embeddings.

        Args:
            model: Trained encoder model
            dataset: Dataset to index
            device: Device for inference
            batch_size: Batch size for embedding extraction
            max_samples: Maximum number of samples to index (None = all)
        """
        model.eval()
        model.to(device)

        # Determine number of samples
        dataset_len = len(dataset)  # type: ignore
        n_samples = dataset_len
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        # Create temporary dataloader
        indices = np.random.permutation(dataset_len)[:n_samples].tolist()
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []
        all_images = []

        with torch.no_grad():
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(device)

                outputs = model(batch_images)
                embeddings = outputs["embedding"].cpu().numpy()

                all_embeddings.append(embeddings)
                all_labels.append(batch_labels.numpy().flatten())

                # Store denormalized images for display
                denorm_images = inverse_normalize(batch_images.cpu())
                # Convert to numpy HWC format
                images_np = denorm_images.permute(0, 2, 3, 1).numpy()
                images_np = np.clip(images_np * 255, 0, 255).astype(np.uint8)
                all_images.append(images_np)

        # Concatenate all batches
        self.embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.labels = np.concatenate(all_labels)
        self.images = np.vstack(all_images)
        self.sample_indices = indices

        # Build Faiss index
        self.index.reset()
        self.index.add(self.embeddings)

        self.is_built = True
        print(f"Built Faiss index with {len(self.embeddings)} samples")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for similar cases.

        Args:
            query_embedding: Query embedding (1, embedding_dim) or (embedding_dim,)
            top_k: Number of similar cases to return

        Returns:
            Tuple of (similarities, labels, images, indices)
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Search
        similarities, indices = self.index.search(query_embedding, top_k)

        # Flatten results
        similarities = similarities.flatten()
        indices = indices.flatten()

        # Get corresponding labels and images
        retrieved_labels = self.labels[indices]
        retrieved_images = self.images[indices]

        return similarities, retrieved_labels, retrieved_images, indices

    def save(self, save_dir: str) -> None:
        """
        Save the index and metadata to disk.

        Args:
            save_dir: Directory to save files
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save Faiss index
        faiss.write_index(self.index, str(save_path / "faiss_index.bin"))

        # Save metadata
        metadata = {
            "embeddings": self.embeddings,
            "labels": self.labels,
            "images": self.images,
            "sample_indices": self.sample_indices,
            "embedding_dim": self.embedding_dim,
            "is_built": self.is_built,
        }

        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"Saved index to {save_path}")

    def load(self, save_dir: str) -> None:
        """
        Load the index and metadata from disk.

        Args:
            save_dir: Directory containing saved files
        """
        load_path = Path(save_dir)

        # Load Faiss index
        index_file = load_path / "faiss_index.bin"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        self.index = faiss.read_index(str(index_file))

        # Load metadata
        with open(load_path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.embeddings = metadata["embeddings"]
        self.labels = metadata["labels"]
        self.images = metadata["images"]
        self.sample_indices = metadata["sample_indices"]
        self.embedding_dim = metadata["embedding_dim"]
        self.is_built = metadata["is_built"]

        if self.embeddings is not None:
            print(f"Loaded index from {load_path} ({len(self.embeddings)} samples)")
        else:
            print(f"Loaded index from {load_path}")


# ============================================================================
# Causal Counterfactual Analysis (Mock Logic)
# ============================================================================


class CausalCounterfactualAnalyzer:
    """
    Mock causal inference module for demonstrating counterfactual analysis.

    Simulates how changes in clinical features might affect predictions.
    In a real system, this would use actual causal models (e.g., SCM, DoWhy).
    """

    # Mock feature importance for each class
    FEATURE_EFFECTS = {
        # Feature: {class_id: effect_weight}
        "lesion_symmetry": {
            0: -0.15,  # Symmetric lesions less likely to be akiec
            1: -0.20,  # Symmetric less likely BCC
            2: 0.10,  # Symmetric more likely benign
            3: 0.15,  # Symmetric more likely dermatofibroma
            4: -0.35,  # Symmetric much less likely melanoma (asymmetry is key)
            5: 0.20,  # Symmetric more likely nevi
            6: 0.05,  # Slight effect on vascular
        },
        "patient_age": {
            # Effect per 10 years of age increase
            0: 0.08,  # Older -> more likely akiec (sun damage)
            1: 0.10,  # Older -> more likely BCC
            2: 0.05,  # Slight age effect
            3: 0.02,  # Minimal effect
            4: 0.06,  # Some age effect for melanoma
            5: -0.05,  # Younger more likely to have nevi
            6: 0.03,  # Slight effect
        },
        "lesion_border_regularity": {
            0: -0.10,
            1: -0.15,
            2: 0.15,
            3: 0.10,
            4: -0.40,  # Irregular borders strongly indicate melanoma
            5: 0.20,
            6: 0.05,
        },
        "color_uniformity": {
            0: -0.05,
            1: -0.10,
            2: 0.10,
            3: 0.15,
            4: -0.30,  # Multiple colors indicate melanoma
            5: 0.15,
            6: 0.05,
        },
    }

    def __init__(self, num_classes: int = 7):
        """
        Initialize the causal analyzer.

        Args:
            num_classes: Number of classification classes
        """
        self.num_classes = num_classes

    def analyze_counterfactual(
        self, base_probabilities: np.ndarray, feature_name: str, feature_change: float
    ) -> Dict[str, Any]:
        """
        Analyze counterfactual: what if a feature was different?

        Args:
            base_probabilities: Original prediction probabilities (num_classes,)
            feature_name: Name of feature to modify
            feature_change: Amount of change (-1 to 1 for binary, actual value for continuous)

        Returns:
            Dict containing counterfactual analysis results
        """
        if feature_name not in self.FEATURE_EFFECTS:
            available = list(self.FEATURE_EFFECTS.keys())
            raise ValueError(f"Unknown feature: {feature_name}. Available: {available}")

        effects = self.FEATURE_EFFECTS[feature_name]

        # Apply counterfactual effects
        modified_logits = np.log(base_probabilities + 1e-8)

        for class_id, effect_weight in effects.items():
            # Scale effect by feature change magnitude
            if feature_name == "patient_age":
                # Age effect per 10 years
                scaled_effect = effect_weight * (feature_change / 10)
            else:
                # Binary/normalized features
                scaled_effect = effect_weight * feature_change

            modified_logits[class_id] += scaled_effect

        # Convert back to probabilities
        modified_probs = np.exp(modified_logits)
        modified_probs = modified_probs / modified_probs.sum()

        # Calculate change
        prob_changes = modified_probs - base_probabilities

        return {
            "original_probs": base_probabilities,
            "modified_probs": modified_probs,
            "probability_changes": prob_changes,
            "feature_name": feature_name,
            "feature_change": feature_change,
            "most_affected_class": int(np.argmax(np.abs(prob_changes))),
            "interpretation": self._generate_interpretation(
                feature_name, feature_change, prob_changes
            ),
        }

    def _generate_interpretation(
        self, feature_name: str, feature_change: float, prob_changes: np.ndarray
    ) -> str:
        """Generate clinical interpretation of counterfactual."""
        from model import DERMAMNIST_LABELS

        most_increased = int(np.argmax(prob_changes))
        most_decreased = int(np.argmin(prob_changes))

        increase_pct = prob_changes[most_increased] * 100
        decrease_pct = prob_changes[most_decreased] * 100

        feature_desc = {
            "lesion_symmetry": "lesion symmetry",
            "patient_age": "patient age",
            "lesion_border_regularity": "border regularity",
            "color_uniformity": "color uniformity",
        }

        change_desc = "increasing" if feature_change > 0 else "decreasing"

        interpretation = (
            f"If {change_desc} {feature_desc.get(feature_name, feature_name)}:\n"
        )
        interpretation += f"• Probability of {DERMAMNIST_LABELS[most_increased]} increases by {increase_pct:+.1f}%\n"
        interpretation += f"• Probability of {DERMAMNIST_LABELS[most_decreased]} decreases by {abs(decrease_pct):.1f}%"

        return interpretation

    def get_all_counterfactuals(
        self, base_probabilities: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get counterfactual analysis for all features.

        Args:
            base_probabilities: Original prediction probabilities

        Returns:
            Dict mapping feature names to counterfactual results
        """
        results = {}

        feature_changes = {
            "lesion_symmetry": 1.0,  # If lesion becomes symmetric
            "patient_age": 20,  # If patient is 20 years older
            "lesion_border_regularity": 1.0,  # If borders become regular
            "color_uniformity": 1.0,  # If color becomes uniform
        }

        for feature_name, change in feature_changes.items():
            results[feature_name] = self.analyze_counterfactual(
                base_probabilities, feature_name, change
            )

        return results


# ============================================================================
# Visualization Helpers
# ============================================================================


def apply_gradcam_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Apply Grad-CAM heatmap overlay to image.

    Args:
        image: Original image (H, W, 3) uint8
        heatmap: Grad-CAM heatmap (H', W') float
        alpha: Overlay transparency

    Returns:
        Image with heatmap overlay (H, W, 3) uint8
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize heatmap
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (
        heatmap_resized.max() - heatmap_resized.min() + 1e-8
    )

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def preprocess_uploaded_image(
    image: Image.Image, image_size: int = 224
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess an uploaded image for model inference.

    Args:
        image: PIL Image
        image_size: Target size

    Returns:
        Tuple of (preprocessed tensor, display image numpy array)
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize for display
    display_image = image.resize((image_size, image_size))
    display_np = np.array(display_image)

    # Apply transforms for model
    transform = get_transforms(image_size=image_size, training=False)
    tensor: torch.Tensor = transform(image)  # type: ignore
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    return tensor, display_np


def generate_clinical_recommendation(
    predicted_class: int, confidence: float, similar_cases_labels: np.ndarray
) -> str:
    """
    Generate clinical recommendation based on prediction and similar cases.

    Args:
        predicted_class: Predicted class index
        confidence: Prediction confidence (0-1)
        similar_cases_labels: Labels of retrieved similar cases

    Returns:
        Clinical recommendation text
    """
    from model import CLINICAL_DESCRIPTIONS, DERMAMNIST_LABELS, RISK_LEVELS

    label = DERMAMNIST_LABELS[predicted_class]
    description = CLINICAL_DESCRIPTIONS[predicted_class]
    risk = RISK_LEVELS[predicted_class]

    # Check consistency with similar cases
    similar_match = (similar_cases_labels == predicted_class).mean()

    recommendation = f"## AI Analysis Summary\n\n"
    recommendation += f"**Predicted Diagnosis:** {label}\n\n"
    recommendation += f"**Confidence Level:** {confidence * 100:.1f}%\n\n"
    recommendation += f"**Risk Assessment:** {risk}\n\n"
    recommendation += f"**Clinical Context:** {description}\n\n"

    # Consistency analysis
    recommendation += f"### Similar Case Analysis\n\n"
    recommendation += f"Of the retrieved similar cases, {similar_match * 100:.0f}% share the same diagnosis.\n\n"

    if similar_match < 0.5:
        recommendation += "⚠️ **Note:** Low agreement with similar cases suggests additional review may be warranted.\n\n"

    # Risk-specific recommendations
    if risk == "CRITICAL":
        recommendation += "### 🚨 Urgent Recommendations\n\n"
        recommendation += "1. **Immediate dermatologist consultation** recommended\n"
        recommendation += "2. Consider **biopsy** for histopathological confirmation\n"
        recommendation += "3. Document lesion with **dermoscopy** if available\n"
        recommendation += "4. Assess for **regional lymphadenopathy**\n"
    elif risk == "HIGH":
        recommendation += "### ⚠️ Action Items\n\n"
        recommendation += "1. Schedule **dermatologist appointment** within 2 weeks\n"
        recommendation += "2. Consider **excisional biopsy** for definitive diagnosis\n"
        recommendation += "3. Review patient's sun exposure history\n"
    elif risk == "MODERATE":
        recommendation += "### 📋 Follow-up Actions\n\n"
        recommendation += "1. Monitor lesion for changes using **ABCDE criteria**\n"
        recommendation += "2. Consider **cryotherapy** or topical treatment\n"
        recommendation += "3. Schedule follow-up in **1-3 months**\n"
    else:
        recommendation += "### ✅ Management Notes\n\n"
        recommendation += "1. Lesion appears benign based on AI analysis\n"
        recommendation += "2. **Routine monitoring** recommended\n"
        recommendation += "3. Patient education on skin self-examination\n"

    recommendation += "\n---\n"
    recommendation += "*⚕️ This AI system is for decision support only. "
    recommendation += "Clinical judgment and histopathological confirmation are essential for diagnosis.*"

    return recommendation


# ============================================================================
# Model and Index Persistence
# ============================================================================


def get_model_path(models_dir: str = "./models") -> Path:
    """Get default model checkpoint path."""
    return Path(models_dir) / "cs_mdss_checkpoint.ckpt"


def get_index_path(models_dir: str = "./models") -> Path:
    """Get default index directory path."""
    return Path(models_dir) / "faiss_index"


def check_resources_exist(models_dir: str = "./models") -> Dict[str, Union[bool, str]]:
    """
    Check if model and index files exist.

    Args:
        models_dir: Models directory

    Returns:
        Dict indicating existence of resources
    """
    model_path = get_model_path(models_dir)
    index_path = get_index_path(models_dir)

    return {
        "model_exists": model_path.exists(),
        "index_exists": (index_path / "faiss_index.bin").exists(),
        "model_path": str(model_path),
        "index_path": str(index_path),
    }
