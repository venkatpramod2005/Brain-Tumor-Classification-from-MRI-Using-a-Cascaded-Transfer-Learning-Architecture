# -*- coding: utf-8 -*-
"""
Brain Tumor Detection - Streamlit Web Application
Beautiful, modern UI for MRI-based brain tumor classification
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import time

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================

st.markdown("""
<style>
    :root {
        --primary-color: #1f8f4d;
        --primary-strong: #176d3b;
        --success-color: #1f8f4d;
        --warning-color: #b87a1f;
        --danger-color: #bb3f3f;
        --app-bg: #f3f6f4;
        --surface-bg: #ffffff;
        --surface-alt: #eef3ef;
        --text-color: #172219;
        --muted-text: #5b675e;
        --border-color: #d7e1d9;
        --shadow-color: rgba(18, 28, 21, 0.10);
        --header-start: #1b5e20;
        --header-end: #0d3b0d;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #4bc07d;
            --primary-strong: #29945d;
            --success-color: #4bc07d;
            --warning-color: #d7a64f;
            --danger-color: #e06f6f;
            --app-bg: #0f1411;
            --surface-bg: #151d17;
            --surface-alt: #1c2720;
            --text-color: #e6efe8;
            --muted-text: #a6b5aa;
            --border-color: #2c3a30;
            --shadow-color: rgba(0, 0, 0, 0.32);
            --header-start: #1f6b36;
            --header-end: #113a22;
        }
    }

    html, body, .stApp {
        background-color: var(--app-bg);
        color: var(--text-color);
        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: var(--surface-bg);
        border-right: 1px solid var(--border-color);
    }

    .main-header {
        background: linear-gradient(135deg, var(--header-start) 0%, var(--header-end) 100%);
        padding: 2rem;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 10px var(--shadow-color);
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.92;
    }

    .result-card {
        background: var(--surface-bg);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px var(--shadow-color);
        margin-bottom: 1rem;
        border-left: 5px solid var(--primary-color);
    }

    .prediction-box {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-strong) 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 10px var(--shadow-color);
    }

    .prediction-box h2 {
        font-size: 2rem;
        margin: 0;
    }

    .metric-card {
        background: var(--surface-alt);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .metric-value {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.45rem 0;
    }

    .metric-label {
        color: var(--muted-text);
        font-size: 0.9rem;
    }

    .upload-text {
        text-align: center;
        color: var(--muted-text);
        padding: 2rem;
    }

    .info-box,
    .warning-box,
    .success-box {
        background: var(--surface-alt);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        border-left: 4px solid var(--primary-color);
    }

    .warning-box {
        border-left-color: var(--warning-color);
    }

    .footer {
        text-align: center;
        color: var(--muted-text);
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border-color);
    }

    .stButton>button {
        background-color: var(--primary-color);
        color: #ffffff;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .stButton>button:hover {
        background-color: var(--primary-strong);
        box-shadow: 0 4px 8px var(--shadow-color);
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--surface-bg);
        border: 1px dashed var(--border-color);
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: var(--muted-text) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"

# Model paths
BINARY_MODEL_PATH = MODELS_DIR / "best_model_binary_ResNet50_20260331_202827.keras"
MULTICLASS_MODEL_PATH = MODELS_DIR / "best_model_mc.keras"

# Class names for different models
BINARY_CLASS_NAMES = ['No Tumor', 'Tumor']
MULTICLASS_CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary']

CLASS_DESCRIPTIONS = {
    'Glioma': 'A tumor that develops from glial cells in the brain or spine',
    'Meningioma': 'A tumor that forms on membranes covering the brain and spinal cord',
    'Pituitary': 'A tumor in the pituitary gland at the base of the brain',
    'Tumor': 'Brain tumor detected - requires further classification',
    'No Tumor': 'No brain tumor detected in the MRI scan'
}

# MC Dropout settings
DEFAULT_MC_PASSES = 20
VARIANCE_THRESHOLDS = {
    'low': 0.01,
    'medium': 0.05
}

# ==================== MODEL LOADING ====================

@st.cache_resource
def load_models():
    """Load both binary and multi-class models with caching."""
    models = {}
    errors = {}
    
    # Load binary model
    try:
        models['binary'] = keras.models.load_model(BINARY_MODEL_PATH)
    except Exception as e:
        errors['binary'] = str(e)
    
    # Load multi-class model
    try:
        models['multiclass'] = keras.models.load_model(MULTICLASS_MODEL_PATH)
    except Exception as e:
        errors['multiclass'] = str(e)
    
    return models, errors if errors else None

# ==================== PREPROCESSING ====================

def preprocess_image(image):
    """
    Preprocess image for model prediction.
    Matches training preprocessing pipeline.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to array
    img_array = np.array(image, dtype=np.float32)
    
    # Apply ResNet50 preprocessing
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

# ==================== PREDICTION FUNCTIONS ====================

def predict_standard(model, image, is_binary=False):
    """Standard prediction without MC Dropout."""
    pred = model.predict(image, verbose=0)
    
    if is_binary:
        # Binary classification (sigmoid output)
        tumor_prob = float(pred[0][0])  # Probability of tumor (class 1)
        predicted_class = 1 if tumor_prob > 0.5 else 0
        
        # Confidence is the probability of the PREDICTED class
        if predicted_class == 1:
            confidence = tumor_prob  # Tumor detected
        else:
            confidence = 1 - tumor_prob  # No tumor detected
        
        # Create probability array [no_tumor, tumor]
        probabilities = np.array([1 - tumor_prob, tumor_prob])
    else:
        # Multi-class (softmax output)
        predicted_class = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))
        probabilities = pred[0]
    
    # Entropy-based uncertainty
    epsilon = 1e-10
    entropy = -np.sum(probabilities * np.log2(probabilities + epsilon))
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'entropy': float(entropy),
        'method': 'Standard'
    }

def predict_mc_dropout(model, image, n_passes=20, is_binary=False):
    """MC Dropout prediction with uncertainty estimation."""
    predictions = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_passes):
        pred = model(image, training=True).numpy()
        
        if is_binary:
            # Convert sigmoid output to probability array [no_tumor, tumor]
            tumor_prob = float(pred[0][0])
            pred_array = np.array([1 - tumor_prob, tumor_prob])
        else:
            pred_array = pred[0]
        
        predictions.append(pred_array)
        
        # Update progress
        progress = (i + 1) / n_passes
        progress_bar.progress(progress)
        status_text.text(f"MC Dropout: Pass {i + 1}/{n_passes}")
    
    progress_bar.empty()
    status_text.empty()
    
    predictions = np.array(predictions)
    
    # Compute statistics
    mean_pred = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    mean_variance = np.mean(variance)
    
    predicted_class = int(np.argmax(mean_pred))
    # Confidence is the probability of the PREDICTED class
    confidence = float(mean_pred[predicted_class])
    
    # Entropy
    epsilon = 1e-10
    entropy = -np.sum(mean_pred * np.log2(mean_pred + epsilon))
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': mean_pred,
        'entropy': float(entropy),
        'variance': variance,
        'mean_variance': float(mean_variance),
        'all_predictions': predictions,
        'method': 'MC Dropout',
        'n_passes': n_passes
    }

def classify_uncertainty(result, use_mc_dropout=False):
    """Classify prediction uncertainty."""
    if use_mc_dropout and 'mean_variance' in result:
        variance = result['mean_variance']
        if variance < VARIANCE_THRESHOLDS['low']:
            return 'High Confidence', 'success', 'Reliable prediction'
        elif variance < VARIANCE_THRESHOLDS['medium']:
            return 'Medium Confidence', 'warning', 'Moderate uncertainty'
        else:
            return 'Low Confidence', 'error', 'High uncertainty - review recommended'
    else:
        # Entropy-based
        confidence = result['confidence']
        if confidence > 0.8:
            return 'High Confidence', 'success', 'Reliable prediction'
        elif confidence > 0.5:
            return 'Medium Confidence', 'warning', 'Moderate confidence'
        else:
            return 'Low Confidence', 'error', 'Low confidence - review recommended'

# ==================== VISUALIZATION FUNCTIONS ====================

def create_probability_chart(probabilities, class_names):
    """Create interactive probability bar chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities,
            marker_color=['#27AE60', '#E74C3C', '#1B5E20'],
            text=[f'{p:.2%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Class Probabilities',
        xaxis_title='Tumor Type',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create confidence gauge chart."""
    # Determine color
    if confidence > 0.8:
        color = '#27AE60'
    elif confidence > 0.5:
        color = '#F39C12'
    else:
        color = '#E74C3C'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "rgba(231, 76, 60, 0.2)"},
                {'range': [50, 80], 'color': "rgba(243, 156, 18, 0.2)"},
                {'range': [80, 100], 'color': "rgba(46, 204, 113, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_mc_distribution_plot(predictions, class_names):
    """Create box plot showing MC Dropout prediction distribution."""
    import pandas as pd
    
    # Reshape data for plotly
    data = []
    for i, class_name in enumerate(class_names):
        for pred in predictions[:, i]:
            data.append({'Tumor Type': class_name, 'Probability': pred})
    
    df = pd.DataFrame(data)
    
    fig = px.box(df, x='Tumor Type', y='Probability', 
                 color='Tumor Type',
                 color_discrete_sequence=['#27AE60', '#E74C3C', '#1B5E20'])
    
    fig.update_layout(
        title=f'MC Dropout Prediction Distribution ({predictions.shape[0]} passes)',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Brain Tumor Detection System</h1>
        <p>AI-powered MRI classification using deep learning</p>
        <p style="font-size: 1rem; margin-top: 0.5rem;">Dual-stage workflow: Detection to Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models, errors = load_models()
    
    if errors:
        st.error(f"Error loading models: {errors}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        classification_mode = st.radio(
            "Classification Mode",
            options=['Binary (Tumor Detection)', 'Multi-Class (Tumor Type)', 'Two-Stage (Both)'],
            index=2,
            help="Choose classification approach"
        )
        
        use_mc_dropout = st.checkbox(
            "Use MC Dropout",
            value=False,
            help="More accurate uncertainty estimation (slower)"
        )
        
        if use_mc_dropout:
            n_passes = st.slider(
                "Number of MC passes",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="More passes = more accurate but slower"
            )
        else:
            n_passes = DEFAULT_MC_PASSES
        
        st.markdown("---")
        
        st.header("Model Information")
        
        if 'Binary' in classification_mode:
            st.markdown("""
            **Binary Model:**
            - Architecture: ResNet50
            - Accuracy: 95.80%
            - ROC AUC: 0.9882
            - Classes: Tumor / No Tumor
            """)
        
        if 'Multi-Class' in classification_mode or 'Two-Stage' in classification_mode:
            st.markdown("""
            **Multi-Class Model:**
            - Architecture: ResNet50
            - Accuracy: 84.11%
            - Classes: Glioma, Meningioma, Pituitary
            """)
        
        st.markdown("---")
        
        st.header("About")
        
        if 'Binary' in classification_mode:
            st.markdown("""
            **Binary Classification:**
            - Detects presence of tumor
            - High sensitivity (96.36%)
            - First stage screening
            """)
        elif 'Multi-Class' in classification_mode:
            st.markdown("""
            **Tumor Classification:**
            - **Glioma:** Glial cell tumors
            - **Meningioma:** Meningeal tumors  
            - **Pituitary:** Pituitary gland tumors
            """)
        else:  # Two-Stage
            st.markdown("""
            **Two-Stage Classification:**
            1. **Detection:** Is there a tumor?
            2. **Classification:** What type?
            
            **Tumor Types:**
            - Glioma | Meningioma | Pituitary
            """)
        
        st.markdown("---")
        
        # Sample images
        st.header("Sample Images")
        if SAMPLES_DIR.exists():
            sample_files = list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png"))
            if sample_files:
                for sample_file in sample_files[:3]:
                    if st.button(f"Load {sample_file.stem}", key=sample_file.name):
                        st.session_state['sample_image'] = sample_file
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload MRI Image")
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan for analysis"
        )
        
        # Check for sample image
        if 'sample_image' in st.session_state and uploaded_file is None:
            uploaded_file = st.session_state['sample_image']
            st.info(f"Loaded sample: {uploaded_file.name}")
        
        if uploaded_file is not None:
            # Display image
            try:
                if isinstance(uploaded_file, Path):
                    image = Image.open(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                
                # Display image with smaller size
                resized_image = image.copy()
                resized_image.thumbnail((350, 350), Image.Resampling.LANCZOS)
                st.image(resized_image, caption="Uploaded MRI Image", width=350)
                
                # Predict button
                if st.button("Analyze Image", type="primary", width="stretch"):
                    with st.spinner("Processing..."):
                        # Preprocess
                        preprocessed = preprocess_image(image)
                        
                        results = {}
                        tumor_detected = True  # Default for multi-class only mode
                        
                        # Binary classification (Stage 1)
                        if 'Binary' in classification_mode or 'Two-Stage' in classification_mode:
                            with st.spinner("Stage 1: Detecting tumor..."):
                                if use_mc_dropout:
                                    results['binary'] = predict_mc_dropout(
                                        models['binary'], preprocessed, n_passes, is_binary=True
                                    )
                                else:
                                    results['binary'] = predict_standard(
                                        models['binary'], preprocessed, is_binary=True
                                    )
                                
                                # Check if tumor detected (class 1 = Tumor)
                                tumor_detected = results['binary']['predicted_class'] == 1
                        
                        # Multi-class classification (Stage 2) - ONLY if tumor detected or in multi-class mode
                        if 'Multi-Class' in classification_mode:
                            # Multi-class only mode - always classify
                            with st.spinner("Classifying tumor type..."):
                                if use_mc_dropout:
                                    results['multiclass'] = predict_mc_dropout(
                                        models['multiclass'], preprocessed, n_passes, is_binary=False
                                    )
                                else:
                                    results['multiclass'] = predict_standard(
                                        models['multiclass'], preprocessed, is_binary=False
                                    )
                        elif 'Two-Stage' in classification_mode and tumor_detected:
                            # Two-stage mode - only classify if tumor detected
                            with st.spinner("Stage 2: Classifying tumor type..."):
                                if use_mc_dropout:
                                    results['multiclass'] = predict_mc_dropout(
                                        models['multiclass'], preprocessed, n_passes, is_binary=False
                                    )
                                else:
                                    results['multiclass'] = predict_standard(
                                        models['multiclass'], preprocessed, is_binary=False
                                    )
                        
                        # Store in session state
                        st.session_state['results'] = results
                        st.session_state['classification_mode'] = classification_mode
                        st.session_state['use_mc_dropout'] = use_mc_dropout
                        st.session_state['tumor_detected'] = tumor_detected
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            st.markdown("""
            <div class="upload-text">
                <h3>Upload an MRI image to get started</h3>
                <p>Supported formats: JPG, JPEG, PNG</p>
                <p>Or select a sample image from the sidebar</p>
            </div>
            """, unsafe_allow_html=True)

        # Move key metrics under the uploaded image to reduce empty space in the left panel
        if 'results' in st.session_state:
            summary_results = st.session_state['results']
            summary_use_mc = st.session_state.get('use_mc_dropout', False)

            st.markdown("---")
            st.subheader("Summary Metrics")

            if 'binary' in summary_results:
                binary_result = summary_results['binary']
                binary_name = BINARY_CLASS_NAMES[binary_result['predicted_class']]

                st.markdown(f"""
                <div class="result-card">
                    <div class="metric-label">Stage 1 Result</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{binary_name}</div>
                </div>
                """, unsafe_allow_html=True)

                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{binary_result['confidence']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with metric_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Entropy</div>
                        <div class="metric-value">{binary_result['entropy']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            if 'multiclass' in summary_results:
                mc_result = summary_results['multiclass']
                mc_name = MULTICLASS_CLASS_NAMES[mc_result['predicted_class']]

                st.markdown(f"""
                <div class="result-card">
                    <div class="metric-label">Stage 2 Result</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{mc_name}</div>
                </div>
                """, unsafe_allow_html=True)

                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{mc_result['confidence']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with metric_col2:
                    metric_label = "Variance" if summary_use_mc and 'mean_variance' in mc_result else "Entropy"
                    metric_value = (
                        f"{mc_result['mean_variance']:.6f}"
                        if summary_use_mc and 'mean_variance' in mc_result
                        else f"{mc_result['entropy']:.4f}"
                    )
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{metric_label}</div>
                        <div class="metric-value">{metric_value}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.header("Results")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            mode = st.session_state.get('classification_mode', 'Two-Stage (Both)')
            use_mc = st.session_state.get('use_mc_dropout', False)
            
            # Binary results
            if 'binary' in results:
                st.subheader("Stage 1: Tumor Detection")
                binary_result = results['binary']
                binary_class = binary_result['predicted_class']
                binary_name = BINARY_CLASS_NAMES[binary_class]
                binary_conf = binary_result['confidence']
                
                # Display binary prediction
                color = '#27AE60' if binary_class == 0 else '#E74C3C'
                st.markdown(f"""
                <div class="prediction-box" style="background: linear-gradient(135deg, {color} 0%, {'#1B5E20' if binary_class == 0 else '#C0392B'} 100%);">
                    <p style="margin: 0; font-size: 1rem; opacity: 0.9;">Detection Result</p>
                    <h2>{binary_name}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Binary uncertainty
                uncertainty_level, uncertainty_color, uncertainty_msg = classify_uncertainty(binary_result, use_mc)
                if uncertainty_color == 'success':
                    st.success(f"**{uncertainty_level}:** {uncertainty_msg}")
                elif uncertainty_color == 'warning':
                    st.warning(f"**{uncertainty_level}:** {uncertainty_msg}")
                else:
                    st.error(f"**{uncertainty_level}:** {uncertainty_msg}")
                
                # Binary probability chart
                st.plotly_chart(
                    create_probability_chart(binary_result['probabilities'], BINARY_CLASS_NAMES),
                    width='stretch'
                )
                
                st.markdown("---")
            
            # Multi-class results (only if tumor detected or in multi-class only mode)
            tumor_detected = st.session_state.get('tumor_detected', True)
            
            if 'multiclass' in results:
                if 'binary' in results:
                    st.subheader("Stage 2: Tumor Classification")
                
                mc_result = results['multiclass']
                mc_class = mc_result['predicted_class']
                mc_name = MULTICLASS_CLASS_NAMES[mc_class]
                mc_conf = mc_result['confidence']
                
                # Display multi-class prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <p style="margin: 0; font-size: 1rem; opacity: 0.9;">Tumor Type</p>
                    <h2>{mc_name}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Multi-class uncertainty
                uncertainty_level, uncertainty_color, uncertainty_msg = classify_uncertainty(mc_result, use_mc)
                if uncertainty_color == 'success':
                    st.success(f"**{uncertainty_level}:** {uncertainty_msg}")
                elif uncertainty_color == 'warning':
                    st.warning(f"**{uncertainty_level}:** {uncertainty_msg}")
                else:
                    st.error(f"**{uncertainty_level}:** {uncertainty_msg}")
                
                # Multi-class probability chart
                st.plotly_chart(
                    create_probability_chart(mc_result['probabilities'], MULTICLASS_CLASS_NAMES),
                    width='stretch'
                )
                
                # MC Dropout distribution
                if use_mc and 'all_predictions' in mc_result:
                    st.plotly_chart(
                        create_mc_distribution_plot(mc_result['all_predictions'], MULTICLASS_CLASS_NAMES),
                        width='stretch'
                    )
                
                # Tumor description
                with st.expander(f"About {mc_name}"):
                    st.markdown(CLASS_DESCRIPTIONS[mc_name])
            
            elif 'binary' in results and not tumor_detected:
                # No tumor detected - skip classification
                st.subheader("Stage 2: Tumor Classification")
                st.info("""
                **Classification Skipped**
                
                No tumor detected in Stage 1, so tumor type classification was not performed.
                
                This saves processing time and provides a cleaner diagnostic workflow.
                """)
            
            # Combined detailed metrics
            with st.expander("Detailed Metrics"):
                if 'binary' in results:
                    st.markdown("**Binary Classification:**")
                    st.markdown(f"""
                    - Method: {results['binary']['method']}
                    - Confidence: {results['binary']['confidence']:.2%}
                    - Entropy: {results['binary']['entropy']:.4f}
                    """)
                    if use_mc and 'mean_variance' in results['binary']:
                        st.markdown(f"- Mean Variance: {results['binary']['mean_variance']:.6f}")
                    st.markdown("")
                
                if 'multiclass' in results:
                    st.markdown("**Multi-Class Classification:**")
                    st.markdown(f"""
                    - Method: {results['multiclass']['method']}
                    - Confidence: {results['multiclass']['confidence']:.2%}
                    - Entropy: {results['multiclass']['entropy']:.4f}
                    """)
                    if use_mc and 'mean_variance' in results['multiclass']:
                        st.markdown(f"- Mean Variance: {results['multiclass']['mean_variance']:.6f}")
                        st.markdown(f"- MC Passes: {results['multiclass']['n_passes']}")
        
        else:
            st.info("Upload an image and click 'Analyze Image' to see results")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Medical Disclaimer</strong></p>
        <p>This system is for research and educational purposes only.</p>
        <p>It is NOT a substitute for professional medical diagnosis.</p>
        <p>Always consult qualified healthcare professionals for medical decisions.</p>
        <hr style="width: 50%; margin: 1rem auto;">
        <p style="font-size: 0.9rem;">Powered by ResNet50 | Built with Streamlit | (c) 2026</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
