"""
CS-MDSS Evaluation Script
=========================
Run test evaluation and build Faiss index from an existing checkpoint.

Usage:
    python src/evaluate.py
    python src/evaluate.py --checkpoint models/cs_mdss_checkpoint.ckpt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import DERMAMNIST_LABELS, CSMDSSLightningModule
from utils import (
    SimilarityEngine,
    get_index_path,
    get_model_path,
    get_transforms,
    load_dermamnist_datasets,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate CS-MDSS model and build Faiss index"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detect if not specified)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for dataset storage",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory for saving index",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (28 or 224)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--index-samples",
        type=int,
        default=2000,
        help="Number of samples to index",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test evaluation",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip Faiss index building",
    )

    return parser.parse_args()


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.

    Args:
        model: The encoder model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    model.to(device)

    total_correct = 0
    total_samples = 0
    class_correct = {i: 0 for i in range(7)}
    class_total = {i: 0 for i in range(7)}

    print("\n📊 Evaluating on test set...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.view(-1).long()

            outputs = model(images)
            logits = outputs["logits"]

            preds = logits.argmax(dim=1).cpu()

            # Overall accuracy
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            # Per-class accuracy
            for pred, label in zip(preds, labels):
                label_idx = label.item()
                class_total[label_idx] += 1
                if pred == label:
                    class_correct[label_idx] += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches...")

    # Calculate metrics
    overall_acc = total_correct / total_samples
    per_class_acc = {}
    for i in range(7):
        if class_total[i] > 0:
            per_class_acc[i] = class_correct[i] / class_total[i]
        else:
            per_class_acc[i] = 0.0

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "total_samples": total_samples,
        "class_totals": class_total,
    }


def print_evaluation_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("📈 Test Evaluation Results")
    print("=" * 60)

    print(f"\n✅ Overall Accuracy: {results['overall_accuracy'] * 100:.2f}%")
    print(f"   Total Samples: {results['total_samples']}")

    print("\n📊 Per-Class Accuracy:")
    print("-" * 50)
    for class_idx, acc in results["per_class_accuracy"].items():
        label = DERMAMNIST_LABELS[class_idx]
        count = results["class_totals"][class_idx]
        print(f"   {class_idx}: {label:<35} {acc * 100:>6.2f}% (n={count})")
    print("-" * 50)


def main():
    """Main function."""
    args = parse_args()

    print("=" * 60)
    print("CS-MDSS Evaluation & Index Building")
    print("=" * 60)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = get_model_path(args.output_dir)

    if not checkpoint_path.exists():
        # Try to find any .ckpt file
        ckpt_files = list(Path(args.output_dir).glob("*.ckpt"))
        if ckpt_files:
            checkpoint_path = ckpt_files[0]
        else:
            print(f"❌ Error: No checkpoint found at {checkpoint_path}")
            print("   Please train the model first with: python src/train.py")
            sys.exit(1)

    print(f"\n📂 Loading checkpoint: {checkpoint_path}")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🖥️  Using device: {device}")

    # Load model
    try:
        lightning_module = CSMDSSLightningModule.load_from_checkpoint(
            str(checkpoint_path), map_location=device
        )
        model = lightning_module.model
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Load test dataset
    print(f"\n📦 Loading DermaMNIST dataset (size={args.image_size})...")
    _, _, test_dataset = load_dermamnist_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        download=True,
    )
    print(f"   Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Run test evaluation
    if not args.skip_test:
        results = evaluate_model(model, test_loader, device)
        print_evaluation_results(results)
    else:
        print("\n⏭️  Skipping test evaluation (--skip-test)")

    # Build Faiss index
    if not args.skip_index:
        print("\n🔍 Building Faiss similarity index...")

        embedding_dim = lightning_module.hparams.get("embedding_dim", 128)
        similarity_engine = SimilarityEngine(embedding_dim=embedding_dim)

        similarity_engine.build_index(
            model=model,
            dataset=test_dataset,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.index_samples,
        )

        # Save index
        index_path = get_index_path(args.output_dir)
        similarity_engine.save(str(index_path))
        print(f"✅ Index saved to: {index_path}")
    else:
        print("\n⏭️  Skipping index building (--skip-index)")

    print("\n" + "=" * 60)
    print("🎉 Evaluation complete!")
    print("=" * 60)
    print("\nTo run the demo:")
    print("  uv run streamlit run src/app.py")


if __name__ == "__main__":
    main()
