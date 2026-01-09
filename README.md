# CS-MDSS: Causal-Similarity Medical Decision Support System

<div align="center">

🏥 **An AI-Powered Dermatological Analysis Demo**

*Combining Deep Learning, Metric Learning, and Causal Inference*

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Overview

CS-MDSS is an MVP demo that showcases how AI can assist clinical decision-making in dermatology through three innovative approaches:

1. **Similarity Retrieval (Metric Learning)**: Find similar historical cases using learned embeddings
2. **Causal Inference**: Understand how changes in clinical features affect predictions
3. **Decision Chain Visualization**: Explain AI reasoning through Grad-CAM heatmaps

This system is designed to augment—not replace—clinical expertise by providing transparent, interpretable AI assistance.

## 🎯 Key Features

### 🔬 Multi-Head Deep Learning Architecture
- **ResNet50 Backbone**: Pretrained feature extractor fine-tuned on DermaMNIST
- **Classification Head**: 7-class skin lesion classification
- **Embedding Head**: 128-dimensional vectors for similarity search
- **Metric Learning**: Combined CrossEntropy + TripletMargin + Cosine losses

### 🔍 Similarity Search Engine
- **Faiss-powered retrieval**: Fast approximate nearest neighbor search
- **Top-K similar cases**: Historical case comparison for clinical context
- **Visual similarity**: "Face recognition" paradigm for medical images

### 🧠 Explainability & Causal Analysis
- **Grad-CAM heatmaps**: Visualize which image regions drive predictions
- **Counterfactual analysis**: "What-if" scenarios for clinical features
- **Risk assessment**: Automated severity classification

### 🖥️ Interactive Clinical Interface
- **Streamlit-based UI**: Professional, intuitive interface
- **Real-time analysis**: Instant predictions and explanations
- **Clinical recommendations**: AI-generated decision support

## 📁 Project Structure

```
ws/
├── src/
│   ├── model.py          # Neural network architecture & loss functions
│   ├── utils.py          # Data loading, Faiss index, counterfactual logic
│   ├── train.py          # PyTorch Lightning training script
│   └── app.py            # Streamlit web interface
├── models/               # Saved model checkpoints & Faiss index
├── data/                 # DermaMNIST dataset (auto-downloaded)
├── logs/                 # TensorBoard training logs
├── pyproject.toml        # Dependencies (uv package manager)
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
cd ws

# Install dependencies with uv
uv sync

# Or install specific packages
uv add torch torchvision pytorch-lightning monai medmnist faiss-cpu shap streamlit pandas numpy matplotlib plotly opencv-python seaborn
```

### Training the Model

```bash
# Train with default parameters
uv run python src/train.py

# Train with custom parameters
uv run python src/train.py \
    --batch-size 64 \
    --max-epochs 50 \
    --learning-rate 1e-4 \
    --image-size 224

# Quick test run
uv run python src/train.py --fast-dev-run
```

**Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 32 | Training batch size |
| `--max-epochs` | 30 | Maximum training epochs |
| `--learning-rate` | 1e-4 | Initial learning rate |
| `--image-size` | 224 | Input image size (28 or 224) |
| `--embedding-dim` | 128 | Embedding vector dimension |
| `--ce-weight` | 1.0 | CrossEntropy loss weight |
| `--triplet-weight` | 0.5 | Triplet loss weight |
| `--cosine-weight` | 0.3 | Cosine similarity loss weight |
| `--index-samples` | 2000 | Number of samples to index |

### Running the Demo

```bash
# Start the Streamlit app
uv run streamlit run src/app.py

# Or with custom port
uv run streamlit run src/app.py --server.port 8501
```

The application will open in your browser at `http://localhost:8501`.

## 📊 Dataset

This project uses **DermaMNIST** from the [MedMNIST v2](https://medmnist.com/) collection.

| Class | Label | Description |
|-------|-------|-------------|
| 0 | akiec | Actinic Keratoses |
| 1 | bcc | Basal Cell Carcinoma |
| 2 | bkl | Benign Keratosis |
| 3 | df | Dermatofibroma |
| 4 | mel | Melanoma |
| 5 | nv | Melanocytic Nevi |
| 6 | vasc | Vascular Lesions |

**Dataset Statistics:**
- Training: 7,007 images
- Validation: 1,003 images
- Test: 2,005 images
- Image size: 224×224 (RGB)

## 🔬 Technical Innovation

### 1. Metric Learning for Medical Imaging

Unlike traditional classification, CS-MDSS learns a **semantic embedding space** where similar lesions cluster together. This enables:

- **Few-shot learning**: Recognize rare conditions with limited examples
- **Case-based reasoning**: Explain predictions through similar historical cases
- **Transfer learning**: Embeddings generalize to unseen lesion types

**Loss Function:**
```
L_total = λ_ce × L_CrossEntropy + λ_triplet × L_TripletMargin + λ_cosine × L_CosineSimilarity
```

### 2. Causal Counterfactual Analysis

The system simulates causal interventions to answer "what-if" questions:

- *"What if the lesion were symmetric?"*
- *"How would a 20-year age difference affect the diagnosis?"*
- *"What if the borders were regular?"*

This provides clinicians with intuitive understanding of feature importance.

### 3. Grad-CAM Explainability

Gradient-weighted Class Activation Mapping highlights diagnostically relevant regions:

- **Visual attention**: Where is the model looking?
- **Clinical correlation**: Do highlighted regions match known diagnostic criteria?
- **Trust calibration**: Help clinicians assess AI reliability

## 📈 Model Architecture

```
CSMDSSEncoder
├── Backbone: ResNet50 (pretrained ImageNet)
│   └── Output: 2048-dim feature vector
├── Classification Head
│   ├── Dropout(0.3)
│   ├── Linear(2048 → 512) + ReLU
│   ├── Dropout(0.3)
│   └── Linear(512 → 7)  # 7 classes
└── Embedding Head
    ├── Linear(2048 → 512) + ReLU
    ├── Dropout(0.3)
    ├── Linear(512 → 128)
    └── L2 Normalization  # Unit sphere
```

## 🖥️ User Interface

The Streamlit interface features:

### Sidebar
- Image upload component
- Top-K retrieval slider
- Confidence threshold adjustment
- Grad-CAM toggle
- Causal analysis toggle
- System status indicators

### Main Panel

**Row 1: Similar Cases**
- Query image display
- Top-K similar historical cases
- Similarity scores and confirmed diagnoses

**Row 2: Prediction Analysis**
- Primary diagnosis with confidence
- Risk level assessment
- Probability distribution chart
- Grad-CAM attention heatmap

**Row 3: Decision Support**
- Causal counterfactual explorer
- Clinical recommendations
- Action items based on risk level

## ⚠️ Limitations & Disclaimers

1. **Research Purpose Only**: This system is designed for demonstration and research. It is NOT approved for clinical use.

2. **Dataset Limitations**: DermaMNIST contains processed, standardized images that may not reflect real-world clinical photography.

3. **Causal Analysis**: The counterfactual module uses simplified mock logic for demonstration. Real causal inference requires domain-specific structural causal models.

4. **Model Performance**: Results depend heavily on training data quality and may not generalize to all populations or imaging conditions.

## 🔮 Future Directions

- [ ] Swin Transformer backbone option
- [ ] SHAP integration for feature importance
- [ ] Real causal models (DoWhy, CausalML)
- [ ] Multi-task learning with lesion segmentation
- [ ] Uncertainty quantification (MC Dropout, Ensembles)
- [ ] DICOM image support
- [ ] Clinical validation study

## 📚 References

1. Yang, J., et al. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." *Scientific Data*, 2023.

2. Schroff, F., et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering." *CVPR*, 2015.

3. Selvaraju, R.R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV*, 2017.

4. Pearl, J. "Causality: Models, Reasoning, and Inference." Cambridge University Press, 2009.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">

**CS-MDSS** | Causal-Similarity Medical Decision Support System

*Bridging AI and Clinical Practice through Explainable Machine Learning*

</div>