"""
CS-MDSS Quick Demo Script
=========================
Creates mock model weights and index for UI demonstration
without requiring full model training.

Usage:
    python src/quick_demo.py
    streamlit run src/app.py
"""

import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import DERMAMNIST_LABELS, CSMDSSEncoder, CSMDSSLightningModule


def create_demo_checkpoint(output_dir: str = "./models") -> str:
    """
    Create a demo checkpoint with pretrained backbone but untrained heads.

    Args:
        output_dir: Directory to save the checkpoint

    Returns:
        Path to the saved checkpoint
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Creating demo model checkpoint...")

    # Create lightning module
    lightning_module = CSMDSSLightningModule(
        num_classes=7,
        embedding_dim=128,
        learning_rate=1e-4,
    )

    # Save checkpoint
    checkpoint_path = output_path / "cs_mdss_checkpoint.ckpt"

    # Create a minimal checkpoint
    checkpoint = {
        "state_dict": lightning_module.state_dict(),
        "hyper_parameters": lightning_module.hparams,
        "pytorch-lightning_version": "2.0.0",
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved checkpoint to: {checkpoint_path}")

    return str(checkpoint_path)


def create_demo_faiss_index(
    output_dir: str = "./models",
    num_samples: int = 500,
    embedding_dim: int = 128,
    image_size: int = 224,
) -> str:
    """
    Create a demo Faiss index with synthetic embeddings.

    Args:
        output_dir: Directory to save the index
        num_samples: Number of synthetic samples
        embedding_dim: Dimension of embeddings
        image_size: Size of synthetic images

    Returns:
        Path to the saved index directory
    """
    index_path = Path(output_dir) / "faiss_index"
    index_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating demo Faiss index with {num_samples} synthetic samples...")

    # Create synthetic embeddings (normalized for cosine similarity)
    np.random.seed(42)
    embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create synthetic labels (7 classes)
    labels = np.random.randint(0, 7, size=num_samples)

    # Create synthetic images (colored patches for each class)
    images = np.zeros((num_samples, image_size, image_size, 3), dtype=np.uint8)

    # Color palette for each class (RGB)
    class_colors = {
        0: [200, 150, 150],  # Pinkish - akiec
        1: [180, 180, 180],  # Gray - bcc
        2: [139, 90, 43],  # Brown - bkl
        3: [150, 100, 80],  # Tan - df
        4: [50, 50, 50],  # Dark - mel
        5: [180, 140, 100],  # Light brown - nv
        6: [180, 50, 50],  # Reddish - vasc
    }

    for i in range(num_samples):
        label = labels[i]
        base_color = np.array(class_colors[label], dtype=np.float32)

        # Add some noise and variation
        noise = np.random.randn(image_size, image_size, 3) * 20
        img = base_color + noise

        # Add a circular "lesion" in the center
        center = image_size // 2
        radius = np.random.randint(30, 60)
        y, x = np.ogrid[:image_size, :image_size]
        mask = (x - center) ** 2 + (y - center) ** 2 < radius**2

        # Darken or lighten the center
        if label in [4]:  # Melanoma - darker center
            img[mask] = img[mask] * 0.5
        elif label in [6]:  # Vascular - redder center
            img[mask, 0] = np.clip(img[mask, 0] + 50, 0, 255)
        else:
            img[mask] = img[mask] * 0.8

        images[i] = np.clip(img, 0, 255).astype(np.uint8)

    # Build Faiss index
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    # Save Faiss index
    faiss.write_index(index, str(index_path / "faiss_index.bin"))

    # Save metadata
    metadata = {
        "embeddings": embeddings,
        "labels": labels,
        "images": images,
        "sample_indices": np.arange(num_samples),
        "embedding_dim": embedding_dim,
        "is_built": True,
    }

    with open(index_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"  Saved Faiss index to: {index_path}")

    return str(index_path)


def print_demo_info():
    """Print information about the demo setup."""
    print("\n" + "=" * 60)
    print("CS-MDSS Demo Setup Complete!")
    print("=" * 60)
    print("\nThe following resources have been created:")
    print("  - Demo model checkpoint (pretrained backbone)")
    print("  - Demo Faiss index (synthetic samples)")
    print("\nNote: This is for UI demonstration only.")
    print("For real predictions, train the model with:")
    print("  uv run python src/train.py")
    print("\nTo run the Streamlit demo:")
    print("  uv run streamlit run src/app.py")
    print("=" * 60)
    print("\nClass Labels:")
    for idx, label in DERMAMNIST_LABELS.items():
        print(f"  {idx}: {label}")
    print()


def main():
    """Main function to create demo resources."""
    output_dir = "./models"

    print("=" * 60)
    print("CS-MDSS Quick Demo Setup")
    print("=" * 60)
    print()

    # Create demo checkpoint
    try:
        create_demo_checkpoint(output_dir)
    except Exception as e:
        print(f"Warning: Could not create demo checkpoint: {e}")
        print("The app will use an untrained model.")

    # Create demo Faiss index
    try:
        create_demo_faiss_index(output_dir)
    except Exception as e:
        print(f"Warning: Could not create demo Faiss index: {e}")
        print("Similarity search will be disabled.")

    print_demo_info()


if __name__ == "__main__":
    main()
