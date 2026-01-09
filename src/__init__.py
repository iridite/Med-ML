"""
CS-MDSS: Causal-Similarity Medical Decision Support System
===========================================================

A medical AI demo combining similarity retrieval, causal inference,
and decision chain visualization for dermatological analysis.

Modules:
    - model: Neural network architectures and loss functions
    - utils: Data loading, Faiss index, and helper functions
    - train: Training pipeline with PyTorch Lightning
    - app: Streamlit interactive interface
"""

from .model import (
    CLINICAL_DESCRIPTIONS,
    DERMAMNIST_LABELS,
    RISK_LEVELS,
    CosineSimilarityLoss,
    CSMDSSEncoder,
    CSMDSSLightningModule,
    GradCAM,
    TripletMiningLoss,
    create_model,
    load_model_from_checkpoint,
)
from .utils import (
    CausalCounterfactualAnalyzer,
    SimilarityEngine,
    apply_gradcam_heatmap,
    check_resources_exist,
    create_dataloaders,
    generate_clinical_recommendation,
    get_index_path,
    get_model_path,
    get_transforms,
    inverse_normalize,
    load_dermamnist_datasets,
    preprocess_uploaded_image,
)

__version__ = "0.1.0"
__author__ = "Medical AI Research"

__all__ = [
    # Model components
    "CSMDSSEncoder",
    "CSMDSSLightningModule",
    "GradCAM",
    "TripletMiningLoss",
    "CosineSimilarityLoss",
    # Labels and descriptions
    "DERMAMNIST_LABELS",
    "CLINICAL_DESCRIPTIONS",
    "RISK_LEVELS",
    # Model utilities
    "load_model_from_checkpoint",
    "create_model",
    # Data and search
    "SimilarityEngine",
    "CausalCounterfactualAnalyzer",
    "get_transforms",
    "inverse_normalize",
    "load_dermamnist_datasets",
    "create_dataloaders",
    # Visualization
    "apply_gradcam_heatmap",
    "preprocess_uploaded_image",
    "generate_clinical_recommendation",
    # Paths
    "get_model_path",
    "get_index_path",
    "check_resources_exist",
]
