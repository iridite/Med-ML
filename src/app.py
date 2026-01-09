"""
CS-MDSS: Causal-Similarity Medical Decision Support System
==========================================================
Streamlit-based interactive demo for medical image analysis.

Features:
- Image upload and preprocessing
- AI-powered skin lesion classification
- Similar case retrieval (Top-K)
- Grad-CAM visualization
- Causal counterfactual analysis
- Clinical decision support recommendations
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import (
    CLINICAL_DESCRIPTIONS,
    DERMAMNIST_LABELS,
    RISK_LEVELS,
    CSMDSSEncoder,
    CSMDSSLightningModule,
    GradCAM,
)
from utils import (
    CausalCounterfactualAnalyzer,
    SimilarityEngine,
    apply_gradcam_heatmap,
    check_resources_exist,
    generate_clinical_recommendation,
    get_index_path,
    get_model_path,
    preprocess_uploaded_image,
)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="CS-MDSS | Medical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional aesthetics
st.markdown(
    """
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #43A047;
        --warning-color: #FB8C00;
        --danger-color: #E53935;
        --background-color: #F5F7FA;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }

    /* Card styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Risk level badges */
    .risk-critical {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
    }

    .risk-high {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
    }

    .risk-moderate {
        background-color: #FFF8E1;
        color: #F57F17;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
    }

    .risk-low {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
    }

    /* Similar case card */
    .similar-case {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
        background: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #E0E0E0;
        margin-top: 2rem;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# Session State Initialization
# ============================================================================


def init_session_state():
    """Initialize session state variables."""
    if "model" not in st.session_state:
        st.session_state.model = None
    if "similarity_engine" not in st.session_state:
        st.session_state.similarity_engine = None
    if "gradcam" not in st.session_state:
        st.session_state.gradcam = None
    if "causal_analyzer" not in st.session_state:
        st.session_state.causal_analyzer = CausalCounterfactualAnalyzer()
    if "device" not in st.session_state:
        st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False


# ============================================================================
# Model Loading
# ============================================================================


@st.cache_resource
def load_model(model_path: str, device: str):
    """Load the trained model (cached)."""
    try:
        lightning_module = CSMDSSLightningModule.load_from_checkpoint(
            model_path, map_location=device
        )
        model = lightning_module.model
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_resource
def load_similarity_engine(index_path: str):
    """Load the similarity engine (cached)."""
    try:
        engine = SimilarityEngine()
        engine.load(index_path)
        return engine
    except Exception as e:
        st.error(f"Failed to load similarity index: {e}")
        return None


def create_demo_model():
    """Create a demo model without trained weights for UI testing."""
    model = CSMDSSEncoder(num_classes=7, embedding_dim=128, pretrained=True)
    model.eval()
    return model


# ============================================================================
# Sidebar
# ============================================================================


def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.markdown("## 🏥 CS-MDSS")
    st.sidebar.markdown("**Causal-Similarity Medical Decision Support System**")
    st.sidebar.markdown("---")

    # Image upload
    st.sidebar.markdown("### 📤 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a skin lesion image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, JPEG, PNG, BMP",
    )

    st.sidebar.markdown("---")

    # Parameters
    st.sidebar.markdown("### ⚙️ Parameters")

    top_k = st.sidebar.slider(
        "Top-K Similar Cases",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of similar historical cases to retrieve",
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for primary diagnosis",
    )

    show_gradcam = st.sidebar.checkbox(
        "Show Grad-CAM Heatmap",
        value=True,
        help="Visualize which regions influenced the prediction",
    )

    show_counterfactual = st.sidebar.checkbox(
        "Show Causal Analysis",
        value=True,
        help="Display counterfactual 'what-if' analysis",
    )

    st.sidebar.markdown("---")

    # Model status
    st.sidebar.markdown("### 📊 System Status")

    resources = check_resources_exist("./models")

    if resources["model_exists"]:
        st.sidebar.success("✅ Model loaded")
    else:
        st.sidebar.warning("⚠️ Model not found (using demo mode)")

    if resources["index_exists"]:
        st.sidebar.success("✅ Similarity index loaded")
    else:
        st.sidebar.warning("⚠️ Index not found (similarity search disabled)")

    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"🖥️ Running on: {device}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    ### ℹ️ About
    This demo showcases AI-assisted dermatological diagnosis using:
    - **Metric Learning** for case similarity
    - **Grad-CAM** for explainability
    - **Causal Analysis** for decision support

    *For research purposes only.*
    """
    )

    return {
        "uploaded_file": uploaded_file,
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
        "show_gradcam": show_gradcam,
        "show_counterfactual": show_counterfactual,
    }


# ============================================================================
# Main Content
# ============================================================================


def render_header():
    """Render the main header."""
    st.markdown(
        """
    <div class="main-header">
        <h1>🏥 CS-MDSS</h1>
        <p>Causal-Similarity Medical Decision Support System</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Combining Deep Learning, Metric Learning, and Causal Inference for Dermatological Analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_upload_prompt():
    """Render the upload prompt when no image is uploaded."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
        ### 📤 Upload a Skin Lesion Image

        To get started, please upload a dermoscopic or clinical image of a skin lesion
        using the sidebar on the left.

        **Supported diagnoses:**
        - Actinic Keratoses
        - Basal Cell Carcinoma
        - Benign Keratosis
        - Dermatofibroma
        - Melanoma
        - Melanocytic Nevi
        - Vascular Lesions

        ---

        **Note:** This system is for research and educational purposes only.
        Always consult a qualified healthcare professional for medical advice.
        """
        )

        # Show sample analysis option
        if st.button(
            "🔬 Run Demo Analysis", help="Use a sample image for demonstration"
        ):
            st.session_state["run_demo"] = True
            st.rerun()


def render_prediction_results(
    image_np: np.ndarray,
    probabilities: np.ndarray,
    predicted_class: int,
    confidence: float,
):
    """Render the prediction results section."""
    st.markdown("### 📊 AI Prediction Results")

    col1, col2, col3, col4 = st.columns(4)

    # Primary diagnosis
    with col1:
        st.markdown("**Primary Diagnosis**")
        label = DERMAMNIST_LABELS[predicted_class]
        st.markdown(f"### {label}")

    # Confidence
    with col2:
        st.markdown("**Confidence**")
        color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
        st.markdown(
            f"### <span style='color:{color}'>{confidence * 100:.1f}%</span>",
            unsafe_allow_html=True,
        )

    # Risk level
    with col3:
        st.markdown("**Risk Level**")
        risk = RISK_LEVELS[predicted_class]
        risk_color = {
            "CRITICAL": "risk-critical",
            "HIGH": "risk-high",
            "MODERATE": "risk-moderate",
            "LOW": "risk-low",
        }
        st.markdown(
            f"<span class='{risk_color[risk]}'>{risk}</span>", unsafe_allow_html=True
        )

    # Second opinion
    with col4:
        sorted_indices = np.argsort(probabilities)[::-1]
        if len(sorted_indices) > 1:
            second_class = sorted_indices[1]
            second_prob = probabilities[second_class]
            st.markdown("**Alternative Diagnosis**")
            st.markdown(f"{DERMAMNIST_LABELS[second_class]}")
            st.markdown(f"({second_prob * 100:.1f}%)")


def render_probability_chart(probabilities: np.ndarray):
    """Render probability distribution chart."""
    import plotly.graph_objects as go

    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_labels = [DERMAMNIST_LABELS[i] for i in sorted_indices]

    # Create color scale based on risk
    colors = []
    for idx in sorted_indices:
        risk = RISK_LEVELS[idx]
        if risk == "CRITICAL":
            colors.append("#E53935")
        elif risk == "HIGH":
            colors.append("#FB8C00")
        elif risk == "MODERATE":
            colors.append("#FDD835")
        else:
            colors.append("#43A047")

    fig = go.Figure(
        data=[
            go.Bar(
                x=sorted_probs * 100,
                y=sorted_labels,
                orientation="h",
                marker_color=colors,
                text=[f"{p * 100:.1f}%" for p in sorted_probs],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Prediction Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(range=[0, 105]),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_similar_cases(
    query_image: np.ndarray,
    similarities: np.ndarray,
    labels: np.ndarray,
    images: np.ndarray,
):
    """Render similar cases section."""
    st.markdown("### 🔍 Similar Historical Cases")
    st.markdown(
        "*These cases from the database have similar visual features to the uploaded image.*"
    )

    n_cases = len(similarities)
    cols = st.columns(n_cases + 1)

    # Query image
    with cols[0]:
        st.markdown("**Query Image**")
        st.image(query_image, use_container_width=True)
        st.markdown("*Uploaded*")

    # Similar cases
    for i, col in enumerate(cols[1:]):
        with col:
            similarity_pct = similarities[i] * 100
            label = DERMAMNIST_LABELS[int(labels[i])]

            st.markdown(f"**Match #{i + 1}**")
            st.image(images[i], use_container_width=True)
            st.markdown(f"**{label}**")
            st.markdown(f"Similarity: {similarity_pct:.1f}%")


def render_gradcam_visualization(
    original_image: np.ndarray, heatmap: np.ndarray, predicted_class: int
):
    """Render Grad-CAM visualization section."""
    st.markdown("### 🔥 Attention Heatmap (Grad-CAM)")
    st.markdown(
        "*This heatmap shows which regions of the image most influenced the AI's prediction.*"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Image**")
        st.image(original_image, use_container_width=True)

    with col2:
        st.markdown("**Attention Heatmap**")
        # Create heatmap visualization
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(heatmap, cmap="jet")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close()

    with col3:
        st.markdown("**Overlay**")
        overlay = apply_gradcam_heatmap(original_image, heatmap, alpha=0.5)
        st.image(overlay, use_container_width=True)

    # Clinical interpretation
    st.info(
        f"""
    **Clinical Interpretation:**
    The highlighted regions indicate areas that contributed most to the classification
    as **{DERMAMNIST_LABELS[predicted_class]}**.
    High-attention areas (red/yellow) should be examined carefully for characteristic
    features of the predicted condition.
    """
    )


def render_counterfactual_analysis(probabilities: np.ndarray):
    """Render counterfactual analysis section."""
    st.markdown("### 🔮 Causal Counterfactual Analysis")
    st.markdown(
        "*Explore how changes in clinical features might affect the diagnosis.*"
    )

    analyzer = st.session_state.causal_analyzer

    # Feature selection
    col1, col2 = st.columns(2)

    with col1:
        feature = st.selectbox(
            "Select Clinical Feature",
            options=[
                "lesion_symmetry",
                "patient_age",
                "lesion_border_regularity",
                "color_uniformity",
            ],
            format_func=lambda x: {
                "lesion_symmetry": "Lesion Symmetry",
                "patient_age": "Patient Age",
                "lesion_border_regularity": "Border Regularity",
                "color_uniformity": "Color Uniformity",
            }[x],
        )

    with col2:
        if feature == "patient_age":
            change = st.slider("Age Change (years)", -30, 30, 20)
        else:
            change = st.slider(
                "Feature Change",
                -1.0,
                1.0,
                0.5,
                help="Positive = increase feature, Negative = decrease",
            )

    # Perform counterfactual analysis
    result = analyzer.analyze_counterfactual(probabilities, feature, change)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Probabilities**")
        for i, prob in enumerate(result["original_probs"]):
            label = DERMAMNIST_LABELS[i]
            st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

    with col2:
        st.markdown("**Modified Probabilities (Counterfactual)**")
        for i, prob in enumerate(result["modified_probs"]):
            label = DERMAMNIST_LABELS[i]
            change_val = result["probability_changes"][i] * 100
            change_str = f" ({change_val:+.1f}%)" if abs(change_val) > 0.1 else ""
            st.progress(prob, text=f"{label}: {prob * 100:.1f}%{change_str}")

    # Interpretation
    st.info(f"**Interpretation:**\n{result['interpretation']}")


def render_clinical_recommendation(
    predicted_class: int, confidence: float, similar_labels: np.ndarray
):
    """Render clinical recommendation section."""
    st.markdown("### 📋 Clinical Decision Support")

    recommendation = generate_clinical_recommendation(
        predicted_class, confidence, similar_labels
    )

    st.markdown(recommendation)


def render_footer():
    """Render the footer."""
    st.markdown(
        """
    <div class="footer">
        <p>
            <strong>CS-MDSS</strong> - Causal-Similarity Medical Decision Support System<br>
            Developed for research and educational purposes | Not for clinical use<br>
            © 2024 Medical AI Research
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Main Application
# ============================================================================


def main():
    """Main application entry point."""
    init_session_state()
    render_header()

    # Render sidebar and get parameters
    params = render_sidebar()

    # Check for resources
    resources = check_resources_exist("./models")
    model_exists = bool(resources["model_exists"])
    index_exists = bool(resources["index_exists"])
    model_path = str(resources["model_path"])
    index_path = str(resources["index_path"])

    # Load model if available
    if model_exists and st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_model(model_path, st.session_state.device)
            if st.session_state.model is not None:
                st.session_state.gradcam = GradCAM(st.session_state.model)
                st.session_state.model_loaded = True

    # Load similarity engine if available
    if index_exists and st.session_state.similarity_engine is None:
        with st.spinner("Loading similarity index..."):
            st.session_state.similarity_engine = load_similarity_engine(index_path)

    # Use demo model if no trained model available
    if st.session_state.model is None:
        st.session_state.model = create_demo_model()
        st.session_state.model.to(st.session_state.device)
        st.session_state.gradcam = GradCAM(st.session_state.model)
        st.warning(
            "⚠️ Running in demo mode with untrained model. Results are for UI demonstration only."
        )

    # Handle image upload or demo mode
    uploaded_file = params["uploaded_file"]
    run_demo = st.session_state.get("run_demo", False)

    if uploaded_file is None and not run_demo:
        render_upload_prompt()
    else:
        # Process image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            # Create a demo image (random noise for demonstration)
            st.info("🔬 Running demo analysis with a synthetic image...")
            demo_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            # Add some structure to make it look more realistic
            demo_array = np.clip(demo_array.astype(float) * 0.3 + 100, 0, 255).astype(
                np.uint8
            )
            image = Image.fromarray(demo_array)

        # Preprocess image
        input_tensor, display_image = preprocess_uploaded_image(image, image_size=224)
        input_tensor = input_tensor.to(st.session_state.device)

        # Run inference
        with torch.no_grad():
            outputs = st.session_state.model(input_tensor)
            logits = outputs["logits"]
            embedding = outputs["embedding"]

        probabilities = F.softmax(logits, dim=1).cpu().numpy().flatten()
        predicted_class = int(np.argmax(probabilities))
        confidence = probabilities[predicted_class]

        # Row 1: Uploaded image and similar cases
        st.markdown("---")

        if st.session_state.similarity_engine is not None:
            query_embedding = embedding.cpu().numpy()
            similarities, labels, images, indices = (
                st.session_state.similarity_engine.search(
                    query_embedding, top_k=params["top_k"]
                )
            )
            render_similar_cases(display_image, similarities, labels, images)
            similar_labels = labels
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 📷 Uploaded Image")
                st.image(display_image, use_container_width=True)
            with col2:
                st.info(
                    "Similarity search is disabled. Train the model and build the index to enable this feature."
                )
            similar_labels = np.array([predicted_class])  # Fallback

        # Row 2: Prediction results and probability chart
        st.markdown("---")
        render_prediction_results(
            display_image, probabilities, predicted_class, confidence
        )

        col1, col2 = st.columns(2)
        with col1:
            render_probability_chart(probabilities)
        with col2:
            # Grad-CAM visualization
            if params["show_gradcam"]:
                try:
                    heatmap, _ = st.session_state.gradcam.generate(
                        input_tensor.clone().requires_grad_(True), predicted_class
                    )
                    heatmap_np = heatmap.cpu().numpy()
                    render_gradcam_visualization(
                        display_image, heatmap_np, predicted_class
                    )
                except Exception as e:
                    st.warning(f"Grad-CAM generation failed: {e}")

        # Row 3: Causal analysis and clinical recommendation
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if params["show_counterfactual"]:
                render_counterfactual_analysis(probabilities)

        with col2:
            render_clinical_recommendation(predicted_class, confidence, similar_labels)

    render_footer()


if __name__ == "__main__":
    main()
