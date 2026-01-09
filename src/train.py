"""
CS-MDSS Model Training Script
=============================
Trains the ResNet50-based encoder with multi-task learning:
- Classification (CrossEntropyLoss)
- Metric Learning (TripletMarginLoss + CosineSimilarityLoss)

After training, builds and saves the Faiss index for similarity search.
"""

import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import CSMDSSLightningModule
from utils import (
    SimilarityEngine,
    create_dataloaders,
    get_index_path,
    get_model_path,
    load_dermamnist_datasets,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CS-MDSS model on DermaMNIST dataset"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for dataset storage",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        choices=[28, 224],
        help="Image size (28 or 224)",
    )

    # Model arguments
    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Number of classification classes",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension for metric learning",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Loss weights
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=1.0,
        help="Weight for CrossEntropy loss",
    )
    parser.add_argument(
        "--triplet-weight",
        type=float,
        default=0.5,
        help="Weight for Triplet loss",
    )
    parser.add_argument(
        "--cosine-weight",
        type=float,
        default=0.3,
        help="Weight for Cosine similarity loss",
    )
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=0.3,
        help="Margin for triplet loss",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory for saving models and index",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs",
    )

    # Index building
    parser.add_argument(
        "--build-index",
        action="store_true",
        default=True,
        help="Build Faiss index after training",
    )
    parser.add_argument(
        "--index-samples",
        type=int,
        default=2000,
        help="Number of samples to index (None for all)",
    )

    # Device
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator (auto, cpu, gpu, mps)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Quick test run with 1 batch",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    print("CS-MDSS Training Pipeline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)

    # Load datasets
    print("\n📦 Loading DermaMNIST dataset...")
    train_dataset, val_dataset, test_dataset = load_dermamnist_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        download=True,
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize model
    print("\n🏗️ Initializing CS-MDSS model...")
    model = CSMDSSLightningModule(
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ce_weight=args.ce_weight,
        triplet_weight=args.triplet_weight,
        cosine_weight=args.cosine_weight,
        triplet_margin=args.triplet_margin,
        max_epochs=args.max_epochs,
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="cs_mdss_checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="cs_mdss",
        version="latest",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # Train model
    print("\n🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test model
    print("\n📊 Evaluating on test set...")
    trainer.test(model, test_loader)

    # Load best checkpoint
    best_model_path = callbacks[0].best_model_path
    print(f"\n✅ Best model saved to: {best_model_path}")

    # Build Faiss index
    if args.build_index and not args.fast_dev_run:
        print("\n🔍 Building Faiss similarity index...")

        # Load best model
        best_model = CSMDSSLightningModule.load_from_checkpoint(best_model_path)
        encoder = best_model.model
        encoder.eval()

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder.to(device)

        # Create and build index
        similarity_engine = SimilarityEngine(embedding_dim=args.embedding_dim)
        similarity_engine.build_index(
            model=encoder,
            dataset=test_dataset,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.index_samples,
        )

        # Save index
        index_path = get_index_path(args.output_dir)
        similarity_engine.save(str(index_path))
        print(f"   Index saved to: {index_path}")

    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("=" * 60)
    print(f"\nTo run the demo:")
    print(f"  streamlit run src/app.py")


if __name__ == "__main__":
    main()
