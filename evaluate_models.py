"""
Brain Tumor Classification - Comprehensive Evaluation Pipeline
Evaluates binary and multi-class models without retraining
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve
)
from scipy.stats import entropy

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION ====================

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model paths
BINARY_MODEL_PATH = MODELS_DIR / "best_model_binary_ResNet50_20260331_202827.keras"
MC_MODEL_PATH = MODELS_DIR / "best_model_mc.keras"

# Dataset paths
TEST_DIR = DATASET_DIR / "Testing"

# Constants
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3
INPUT_SHAPE = IMG_SIZE + (IMG_CHANNELS,)

# Class definitions
ALL_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
TUMOR_CLASSES = ['glioma', 'meningioma', 'pituitary']
BINARY_CLASSES = ['notumor', 'tumor']

# Class mappings
CLASS_TO_BINARY = {
    'notumor': 0,
    'glioma': 1,
    'meningioma': 1,
    'pituitary': 1
}

TUMOR_CLASS_TO_IDX = {
    'glioma': 0,
    'meningioma': 1,
    'pituitary': 2
}

# Uncertainty thresholds
UNCERTAINTY_THRESHOLD = 0.5  # Will be updated based on distribution
LOW_CONFIDENCE_THRESHOLD = 0.7

print("=" * 80)
print("Brain Tumor Classification - Evaluation Pipeline")
print("=" * 80)
print(f"\n📁 Base Directory: {BASE_DIR}")
print(f"📁 Dataset Directory: {DATASET_DIR}")
print(f"📁 Models Directory: {MODELS_DIR}")
print(f"📁 Results Directory: {RESULTS_DIR}")
print(f"\n✅ Environment setup complete!")
print("=" * 80)


# ==================== DATA LOADING ====================

def load_test_dataset():
    """
    Load test dataset with true labels.
    Returns images, labels, and file paths.
    """
    print("\n" + "=" * 80)
    print("📂 Loading Test Dataset")
    print("=" * 80)
    
    images = []
    labels = []
    file_paths = []
    
    for class_name in ALL_CLASSES:
        class_dir = TEST_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} not found!")
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        class_count = len(image_files)
        
        print(f"  {class_name:12s}: {class_count:4d} images")
        
        for img_path in image_files:
            images.append(img_path)
            labels.append(class_name)
            file_paths.append(str(img_path))
    
    print(f"\n✅ Total test images loaded: {len(images)}")
    
    return images, labels, file_paths


def prepare_binary_labels(labels: List[str]) -> np.ndarray:
    """Convert class labels to binary (tumor vs notumor)."""
    binary_labels = np.array([CLASS_TO_BINARY[label] for label in labels])
    return binary_labels


def prepare_multiclass_labels(labels: List[str]) -> np.ndarray:
    """Convert tumor class labels to indices (only for tumor images)."""
    mc_labels = np.array([TUMOR_CLASS_TO_IDX[label] for label in labels])
    return mc_labels


def filter_tumor_images(images, labels, file_paths):
    """Filter only tumor images for multi-class evaluation."""
    tumor_mask = np.array([label in TUMOR_CLASSES for label in labels])
    
    tumor_images = [img for img, is_tumor in zip(images, tumor_mask) if is_tumor]
    tumor_labels = [label for label, is_tumor in zip(labels, tumor_mask) if is_tumor]
    tumor_paths = [path for path, is_tumor in zip(file_paths, tumor_mask) if is_tumor]
    
    return tumor_images, tumor_labels, tumor_paths


# ==================== PREPROCESSING ====================

def ensure_rgb(image):
    """Ensure image is RGB (3 channels)."""
    if len(image.shape) == 2:  # Grayscale
        image = tf.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:  # Single channel
        image = tf.concat([image, image, image], axis=-1)
    return image


def load_and_preprocess_image(img_path, preprocess_fn=None):
    """
    Load and preprocess a single image.
    Matches training pipeline: resize -> RGB -> float32 -> backbone preprocessing
    """
    # Load image
    img = keras.preprocessing.image.load_img(str(img_path), target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    
    # Ensure RGB
    img_array = ensure_rgb(img_array)
    
    # Cast to float32
    img_array = tf.cast(img_array, tf.float32)
    
    # Apply backbone-specific preprocessing if provided
    if preprocess_fn is not None:
        img_array = preprocess_fn(img_array)
    
    return img_array.numpy()


def preprocess_batch(image_paths, preprocess_fn=None, batch_size=32):
    """
    Preprocess a batch of images.
    Returns numpy array of preprocessed images.
    """
    print(f"  Preprocessing {len(image_paths)} images...")
    
    preprocessed = []
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i + 1}/{len(image_paths)}")
        
        img_array = load_and_preprocess_image(img_path, preprocess_fn)
        preprocessed.append(img_array)
    
    return np.array(preprocessed)


def get_resnet50_preprocess():
    """Get ResNet50 preprocessing function."""
    return tf.keras.applications.resnet.preprocess_input


def detect_model_backbone(model_path):
    """
    Detect the backbone architecture from model.
    Returns backbone name and preprocessing function.
    """
    # Try to load model info file
    info_path = model_path.parent / "model_info.json"
    
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
            backbone = info.get('backbone', 'Unknown')
            print(f"  📋 Detected backbone from model_info.json: {backbone}")
            
            # Map backbone to preprocessing function
            if backbone == "ResNet50":
                return backbone, tf.keras.applications.resnet.preprocess_input
            elif backbone == "EfficientNetB0":
                return backbone, tf.keras.applications.efficientnet.preprocess_input
            elif backbone == "MobileNetV2":
                return backbone, tf.keras.applications.mobilenet_v2.preprocess_input
    
    # If no info file, inspect model layers
    print("  🔍 Detecting backbone from model layers...")
    model = keras.models.load_model(model_path)
    
    for layer in model.layers:
        layer_name = layer.name.lower()
        if 'resnet' in layer_name:
            print(f"  📋 Detected: ResNet50")
            return "ResNet50", tf.keras.applications.resnet.preprocess_input
        elif 'efficientnet' in layer_name:
            print(f"  📋 Detected: EfficientNetB0")
            return "EfficientNetB0", tf.keras.applications.efficientnet.preprocess_input
        elif 'mobilenet' in layer_name:
            print(f"  📋 Detected: MobileNetV2")
            return "MobileNetV2", tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Default to ResNet50 if cannot detect
    print("  ⚠️  Could not detect backbone, defaulting to ResNet50")
    return "ResNet50", tf.keras.applications.resnet.preprocess_input


# ==================== BINARY MODEL PREDICTION ====================

def evaluate_binary_model(images, labels, file_paths):
    """
    Evaluate binary classification model (tumor vs notumor).
    """
    print("\n" + "=" * 80)
    print("🔬 PHASE 1: Binary Classification (Tumor vs No Tumor)")
    print("=" * 80)
    
    # Load model
    print(f"\n📥 Loading binary model: {BINARY_MODEL_PATH.name}")
    binary_model = keras.models.load_model(BINARY_MODEL_PATH)
    print(f"  ✅ Model loaded successfully")
    print(f"  Input shape: {binary_model.input_shape}")
    print(f"  Output shape: {binary_model.output_shape}")
    
    # Get preprocessing function
    preprocess_fn = get_resnet50_preprocess()
    print(f"  🔧 Using ResNet50 preprocessing")
    
    # Preprocess images
    print(f"\n🔄 Preprocessing images...")
    X_test = preprocess_batch(images, preprocess_fn)
    print(f"  ✅ Preprocessed shape: {X_test.shape}")
    
    # Prepare true labels
    y_true_binary = prepare_binary_labels(labels)
    print(f"  ✅ Labels shape: {y_true_binary.shape}")
    print(f"  Class distribution - No Tumor: {np.sum(y_true_binary == 0)}, Tumor: {np.sum(y_true_binary == 1)}")
    
    # Generate predictions
    print(f"\n🎯 Generating predictions...")
    y_pred_proba = binary_model.predict(X_test, batch_size=32, verbose=1)
    
    # Handle different output shapes
    if y_pred_proba.shape[-1] == 1:  # Binary classification with sigmoid
        y_pred_proba_tumor = y_pred_proba.flatten()
        y_pred_proba_notumor = 1 - y_pred_proba_tumor
    else:  # Binary classification with softmax (2 outputs)
        y_pred_proba_notumor = y_pred_proba[:, 0]
        y_pred_proba_tumor = y_pred_proba[:, 1]
    
    y_pred_binary = (y_pred_proba_tumor > 0.5).astype(int)
    
    print(f"  ✅ Predictions generated")
    print(f"  Predicted - No Tumor: {np.sum(y_pred_binary == 0)}, Tumor: {np.sum(y_pred_binary == 1)}")
    
    # Store results
    results = {
        'model_path': str(BINARY_MODEL_PATH),
        'images': images,
        'file_paths': file_paths,
        'true_labels': labels,
        'y_true': y_true_binary,
        'y_pred': y_pred_binary,
        'y_pred_proba_notumor': y_pred_proba_notumor,
        'y_pred_proba_tumor': y_pred_proba_tumor,
        'confidence_scores': np.maximum(y_pred_proba_notumor, y_pred_proba_tumor)
    }
    
    return results


# ==================== BINARY METRICS ====================

def calculate_binary_metrics(results):
    """Calculate comprehensive metrics for binary classification."""
    print("\n" + "=" * 80)
    print("📊 Binary Classification Metrics")
    print("=" * 80)
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  No Tumor  Tumor")
    print(f"Actual No Tumor      {cm[0, 0]:4d}    {cm[0, 1]:4d}")
    print(f"Actual Tumor         {cm[1, 0]:4d}    {cm[1, 1]:4d}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    target_names = ['No Tumor', 'Tumor']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)
    
    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    
    accuracy = np.mean(y_true == y_pred)
    
    print(f"\n📌 Summary:")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\n  No Tumor - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    print(f"  Tumor    - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
    
    # Store metrics
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'classification_report': report
    }
    
    results['metrics'] = metrics
    return results


# ==================== BINARY VISUALIZATIONS ====================

def visualize_binary_results(results, save_dir):
    """Create visualizations for binary classification."""
    print("\n" + "=" * 80)
    print("📊 Creating Binary Classification Visualizations")
    print("=" * 80)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    y_pred_proba_tumor = results['y_pred_proba_tumor']
    cm = results['metrics']['confusion_matrix']
    
    # 1. Confusion Matrix Heatmap
    print("  📈 Creating confusion matrix heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'],
                cbar_kws={'label': 'Count'})
    plt.title('Binary Classification - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: binary_confusion_matrix.png")
    
    # 2. ROC Curve
    print("  📈 Creating ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba_tumor)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Binary Classification - ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'binary_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: binary_roc_curve.png (AUC = {roc_auc:.4f})")
    
    # 3. Precision-Recall Curve
    print("  📈 Creating precision-recall curve...")
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_pred_proba_tumor)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Binary Classification - Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'binary_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: binary_precision_recall_curve.png (AUC = {pr_auc:.4f})")
    
    # Store AUC scores
    results['metrics']['roc_auc'] = roc_auc
    results['metrics']['pr_auc'] = pr_auc
    
    print(f"\n✅ All binary visualizations saved to: {save_dir}")
    
    return results


# ==================== MULTI-CLASS FILTERING ====================

def filter_for_multiclass(images, labels, file_paths, binary_results=None):
    """
    Filter tumor images for multi-class evaluation.
    Can use either ground truth or binary predictions.
    """
    print("\n" + "=" * 80)
    print("🔍 Filtering Tumor Images for Multi-Class Classification")
    print("=" * 80)
    
    # Filter based on ground truth
    tumor_images, tumor_labels, tumor_paths = filter_tumor_images(images, labels, file_paths)
    
    print(f"\n📊 Ground Truth Tumor Images:")
    print(f"  Total tumor images: {len(tumor_images)}")
    
    # Count per class
    from collections import Counter
    class_counts = Counter(tumor_labels)
    for tumor_class in TUMOR_CLASSES:
        count = class_counts.get(tumor_class, 0)
        print(f"  {tumor_class:12s}: {count:4d} images")
    
    return tumor_images, tumor_labels, tumor_paths


# ==================== MULTI-CLASS MODEL PREDICTION ====================

def evaluate_multiclass_model(tumor_images, tumor_labels, tumor_paths):
    """
    Evaluate multi-class classification model (glioma, meningioma, pituitary).
    """
    print("\n" + "=" * 80)
    print("🔬 PHASE 2: Multi-Class Classification (Tumor Types)")
    print("=" * 80)
    
    # Load model
    print(f"\n📥 Loading multi-class model: {MC_MODEL_PATH.name}")
    mc_model = keras.models.load_model(MC_MODEL_PATH)
    print(f"  ✅ Model loaded successfully")
    print(f"  Input shape: {mc_model.input_shape}")
    print(f"  Output shape: {mc_model.output_shape}")
    
    # Detect backbone and get preprocessing function
    print(f"\n🔍 Detecting model backbone...")
    backbone, preprocess_fn = detect_model_backbone(MC_MODEL_PATH)
    print(f"  🔧 Using {backbone} preprocessing")
    
    # Preprocess images
    print(f"\n🔄 Preprocessing images...")
    X_test_tumor = preprocess_batch(tumor_images, preprocess_fn)
    print(f"  ✅ Preprocessed shape: {X_test_tumor.shape}")
    
    # Prepare true labels
    y_true_mc = prepare_multiclass_labels(tumor_labels)
    print(f"  ✅ Labels shape: {y_true_mc.shape}")
    print(f"  Class distribution:")
    for i, tumor_class in enumerate(TUMOR_CLASSES):
        count = np.sum(y_true_mc == i)
        print(f"    {tumor_class:12s}: {count:4d}")
    
    # Generate predictions
    print(f"\n🎯 Generating predictions...")
    y_pred_proba_mc = mc_model.predict(X_test_tumor, batch_size=32, verbose=1)
    y_pred_mc = np.argmax(y_pred_proba_mc, axis=1)
    
    print(f"  ✅ Predictions generated")
    print(f"  Predicted distribution:")
    for i, tumor_class in enumerate(TUMOR_CLASSES):
        count = np.sum(y_pred_mc == i)
        print(f"    {tumor_class:12s}: {count:4d}")
    
    # Store results
    results = {
        'model_path': str(MC_MODEL_PATH),
        'backbone': backbone,
        'images': tumor_images,
        'file_paths': tumor_paths,
        'true_labels': tumor_labels,
        'y_true': y_true_mc,
        'y_pred': y_pred_mc,
        'y_pred_proba': y_pred_proba_mc,
        'confidence_scores': np.max(y_pred_proba_mc, axis=1)
    }
    
    return results
# Multi-class metrics and visualizations
def calculate_multiclass_metrics(results):
    print("\n" + "=" * 80)
    print("📊 Multi-Class Classification Metrics")
    print("=" * 80)
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"           {'Glioma':>8} {'Mening':>8} {'Pituit':>8}")
    for i, tumor_class in enumerate(TUMOR_CLASSES):
        print(f"Actual {tumor_class[:6]:6s} {cm[i, 0]:8d} {cm[i, 1]:8d} {cm[i, 2]:8d}")
    
    print(f"\n📋 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=TUMOR_CLASSES, digits=4)
    print(report)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2]
    )
    
    accuracy = np.mean(y_true == y_pred)
    
    print(f"\n📌 Summary:")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    for i, tumor_class in enumerate(TUMOR_CLASSES):
        print(f"  {tumor_class:12s} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'classification_report': report
    }
    
    results['metrics'] = metrics
    return results
# Continuation of evaluate_models.py - Remaining functions

def visualize_multiclass_results(results, save_dir):
    """Create visualizations for multi-class classification."""
    print("\n" + "=" * 80)
    print("📊 Creating Multi-Class Classification Visualizations")
    print("=" * 80)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    cm = results['metrics']['confusion_matrix']
    
    # 1. Confusion Matrix Heatmap
    print("  📈 Creating confusion matrix heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=TUMOR_CLASSES,
                yticklabels=TUMOR_CLASSES,
                cbar_kws={'label': 'Count'})
    plt.title('Multi-Class Classification - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: multiclass_confusion_matrix.png")
    
    # 2. Per-class Performance Bar Chart
    print("  📈 Creating per-class performance chart...")
    metrics_df = pd.DataFrame({
        'Class': TUMOR_CLASSES,
        'Precision': results['metrics']['precision'],
        'Recall': results['metrics']['recall'],
        'F1-Score': results['metrics']['f1_score']
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(TUMOR_CLASSES))
    width = 0.25
    
    ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#1f77b4')
    ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#ff7f0e')
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#2ca02c')
    
    ax.set_xlabel('Tumor Class', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Multi-Class Performance by Metric', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TUMOR_CLASSES)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'multiclass_performance_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: multiclass_performance_bars.png")
    
    # 3. Confidence Distribution
    print("  📈 Creating confidence distribution plot...")
    confidence = results['confidence_scores']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall distribution
    axes[0].hist(confidence, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(LOW_CONFIDENCE_THRESHOLD, color='red', linestyle='--',
                    label=f'Low Confidence Threshold ({LOW_CONFIDENCE_THRESHOLD})')
    axes[0].set_xlabel('Confidence Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Per-class distribution
    for i, tumor_class in enumerate(TUMOR_CLASSES):
        class_mask = y_true == i
        class_conf = confidence[class_mask]
        axes[1].hist(class_conf, bins=30, alpha=0.5, label=tumor_class, edgecolor='black')
    
    axes[1].set_xlabel('Confidence Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Confidence Distribution by Class', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'multiclass_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: multiclass_confidence_distribution.png")
    
    print(f"\n✅ All multi-class visualizations saved to: {save_dir}")
    
    return results
# Error Analysis and Uncertainty Functions

def perform_error_analysis(binary_results, mc_results, save_dir):
    """Comprehensive error analysis across both stages."""
    print("\n" + "=" * 80)
    print("🔍 Error Analysis")
    print("=" * 80)
    
    save_dir = Path(save_dir)
    
    # Binary errors
    print("\n📊 Binary Stage Errors:")
    binary_errors = binary_results['y_true'] != binary_results['y_pred']
    n_binary_errors = np.sum(binary_errors)
    print(f"  Total binary errors: {n_binary_errors} / {len(binary_errors)} ({n_binary_errors/len(binary_errors)*100:.2f}%)")
    
    # False positives and negatives
    fp_mask = (binary_results['y_true'] == 0) & (binary_results['y_pred'] == 1)
    fn_mask = (binary_results['y_true'] == 1) & (binary_results['y_pred'] == 0)
    
    n_fp = np.sum(fp_mask)
    n_fn = np.sum(fn_mask)
    
    print(f"  False Positives (No Tumor → Tumor): {n_fp}")
    print(f"  False Negatives (Tumor → No Tumor): {n_fn}")
    
    # Multi-class errors
    print("\n📊 Multi-Class Stage Errors:")
    mc_errors = mc_results['y_true'] != mc_results['y_pred']
    n_mc_errors = np.sum(mc_errors)
    print(f"  Total multi-class errors: {n_mc_errors} / {len(mc_errors)} ({n_mc_errors/len(mc_errors)*100:.2f}%)")
    
    # Most confused pairs
    from itertools import combinations
    confusion_pairs = []
    cm = mc_results['metrics']['confusion_matrix']
    
    for i, j in combinations(range(len(TUMOR_CLASSES)), 2):
        confusion = cm[i, j] + cm[j, i]
        if confusion > 0:
            confusion_pairs.append((TUMOR_CLASSES[i], TUMOR_CLASSES[j], confusion))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\n  Most Confused Class Pairs:")
    for class1, class2, count in confusion_pairs[:5]:
        print(f"    {class1:12s} ↔ {class2:12s}: {count:3d} confusions")
    
    # Low confidence predictions
    low_conf_binary = binary_results['confidence_scores'] < LOW_CONFIDENCE_THRESHOLD
    n_low_conf_binary = np.sum(low_conf_binary)
    
    low_conf_mc = mc_results['confidence_scores'] < LOW_CONFIDENCE_THRESHOLD
    n_low_conf_mc = np.sum(low_conf_mc)
    
    print(f"\n📊 Low Confidence Predictions (<{LOW_CONFIDENCE_THRESHOLD}):")
    print(f"  Binary: {n_low_conf_binary} / {len(low_conf_binary)} ({n_low_conf_binary/len(low_conf_binary)*100:.2f}%)")
    print(f"  Multi-class: {n_low_conf_mc} / {len(low_conf_mc)} ({n_low_conf_mc/len(low_conf_mc)*100:.2f}%)")
    
    # Save error indices
    error_data = {
        'binary_errors': {
            'indices': np.where(binary_errors)[0].tolist(),
            'false_positives': np.where(fp_mask)[0].tolist(),
            'false_negatives': np.where(fn_mask)[0].tolist(),
            'low_confidence': np.where(low_conf_binary)[0].tolist()
        },
        'multiclass_errors': {
            'indices': np.where(mc_errors)[0].tolist(),
            'low_confidence': np.where(low_conf_mc)[0].tolist()
        }
    }
    
    with open(save_dir / 'error_analysis.json', 'w') as f:
        json.dump(error_data, f, indent=2)
    
    print(f"\n✅ Error analysis saved to: {save_dir / 'error_analysis.json'}")
    
    return error_data


def calculate_uncertainty(results, stage='binary'):
    """Calculate prediction uncertainty using entropy."""
    print("\n" + "=" * 80)
    print(f"🎲 Uncertainty Estimation ({stage.capitalize()} Stage)")
    print("=" * 80)
    
    if stage == 'binary':
        # Binary entropy
        p_tumor = results['y_pred_proba_tumor']
        p_notumor = results['y_pred_proba_notumor']
        
        # Avoid log(0)
        p_tumor = np.clip(p_tumor, 1e-10, 1 - 1e-10)
        p_notumor = np.clip(p_notumor, 1e-10, 1 - 1e-10)
        
        H = -(p_tumor * np.log2(p_tumor) + p_notumor * np.log2(p_notumor))
        max_entropy = 1.0  # Binary case
        
    else:  # multiclass
        proba = results['y_pred_proba']
        # Clip to avoid log(0)
        proba = np.clip(proba, 1e-10, 1)
        H = -np.sum(proba * np.log2(proba), axis=1)
        max_entropy = np.log2(len(TUMOR_CLASSES))
    
    # Normalize entropy to [0, 1]
    normalized_H = H / max_entropy
    
    # Determine threshold (using median as default)
    threshold = np.median(normalized_H)
    
    certain_mask = normalized_H < threshold
    uncertain_mask = normalized_H >= threshold
    
    n_certain = np.sum(certain_mask)
    n_uncertain = np.sum(uncertain_mask)
    
    print(f"  Entropy statistics:")
    print(f"    Mean: {np.mean(normalized_H):.4f}")
    print(f"    Median: {np.median(normalized_H):.4f}")
    print(f"    Std: {np.std(normalized_H):.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Certain predictions: {n_certain} ({n_certain/len(H)*100:.2f}%)")
    print(f"  Uncertain predictions: {n_uncertain} ({n_uncertain/len(H)*100:.2f}%)")
    
    # Accuracy stratified by uncertainty
    correct = results['y_true'] == results['y_pred']
    acc_certain = np.mean(correct[certain_mask]) if n_certain > 0 else 0
    acc_uncertain = np.mean(correct[uncertain_mask]) if n_uncertain > 0 else 0
    
    print(f"\n  Accuracy by uncertainty:")
    print(f"    Certain: {acc_certain:.4f} ({acc_certain*100:.2f}%)")
    print(f"    Uncertain: {acc_uncertain:.4f} ({acc_uncertain*100:.2f}%)")
    
    results['uncertainty'] = {
        'entropy': H,
        'normalized_entropy': normalized_H,
        'threshold': threshold,
        'certain_mask': certain_mask,
        'uncertain_mask': uncertain_mask,
        'accuracy_certain': acc_certain,
        'accuracy_uncertain': acc_uncertain
    }
    
    return results
# Main Execution and Reporting Functions

def generate_comprehensive_report(binary_results, mc_results, save_dir):
    """Generate comprehensive evaluation report."""
    print("\n" + "=" * 80)
    print("📝 Generating Comprehensive Report")
    print("=" * 80)
    
    save_dir = Path(save_dir)
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("BRAIN TUMOR CLASSIFICATION - COMPREHENSIVE EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Binary Stage
    report_lines.append("## STAGE 1: BINARY CLASSIFICATION (Tumor vs No Tumor)")
    report_lines.append("-" * 80)
    report_lines.append(f"Model: {binary_results['model_path']}")
    report_lines.append(f"Total Test Images: {len(binary_results['y_true'])}")
    report_lines.append("")
    
    bin_metrics = binary_results['metrics']
    report_lines.append(f"**Overall Accuracy: {bin_metrics['accuracy']:.4f} ({bin_metrics['accuracy']*100:.2f}%)**")
    report_lines.append(f"ROC AUC: {bin_metrics['roc_auc']:.4f}")
    report_lines.append(f"PR AUC: {bin_metrics['pr_auc']:.4f}")
    report_lines.append("")
    
    report_lines.append("### Confusion Matrix:")
    cm = bin_metrics['confusion_matrix']
    report_lines.append(f"```")
    report_lines.append(f"                Predicted")
    report_lines.append(f"              No Tumor  Tumor")
    report_lines.append(f"Actual No Tumor   {cm[0,0]:4d}   {cm[0,1]:4d}")
    report_lines.append(f"Actual Tumor      {cm[1,0]:4d}   {cm[1,1]:4d}")
    report_lines.append(f"```")
    report_lines.append("")
    
    report_lines.append("### Per-Class Metrics:")
    report_lines.append(f"| Class      | Precision | Recall | F1-Score | Support |")
    report_lines.append(f"|------------|-----------|--------|----------|---------|")
    report_lines.append(f"| No Tumor   | {bin_metrics['precision'][0]:.4f}    | {bin_metrics['recall'][0]:.4f} | {bin_metrics['f1_score'][0]:.4f}   | {bin_metrics['support'][0]:4d}    |")
    report_lines.append(f"| Tumor      | {bin_metrics['precision'][1]:.4f}    | {bin_metrics['recall'][1]:.4f} | {bin_metrics['f1_score'][1]:.4f}   | {bin_metrics['support'][1]:4d}    |")
    report_lines.append("")
    report_lines.append("")
    
    # Multi-class Stage
    report_lines.append("## STAGE 2: MULTI-CLASS CLASSIFICATION (Tumor Types)")
    report_lines.append("-" * 80)
    report_lines.append(f"Model: {mc_results['model_path']}")
    report_lines.append(f"Backbone: {mc_results['backbone']}")
    report_lines.append(f"Total Tumor Images: {len(mc_results['y_true'])}")
    report_lines.append("")
    
    mc_metrics = mc_results['metrics']
    report_lines.append(f"**Overall Accuracy: {mc_metrics['accuracy']:.4f} ({mc_metrics['accuracy']*100:.2f}%)**")
    report_lines.append("")
    
    report_lines.append("### Confusion Matrix:")
    cm = mc_metrics['confusion_matrix']
    report_lines.append(f"```")
    report_lines.append(f"              Predicted")
    report_lines.append(f"         Glioma  Mening  Pituit")
    for i, tumor_class in enumerate(TUMOR_CLASSES):
        report_lines.append(f"Actual {tumor_class[:6]:6s}   {cm[i,0]:4d}    {cm[i,1]:4d}    {cm[i,2]:4d}")
    report_lines.append(f"```")
    report_lines.append("")
    
    report_lines.append("### Per-Class Metrics:")
    report_lines.append(f"| Class        | Precision | Recall | F1-Score | Support |")
    report_lines.append(f"|--------------|-----------|--------|----------|---------|")
    for i, tumor_class in enumerate(TUMOR_CLASSES):
        report_lines.append(f"| {tumor_class:12s} | {mc_metrics['precision'][i]:.4f}    | {mc_metrics['recall'][i]:.4f} | {mc_metrics['f1_score'][i]:.4f}   | {mc_metrics['support'][i]:4d}    |")
    report_lines.append("")
    report_lines.append("")
    
    # Key Findings
    report_lines.append("## KEY FINDINGS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    report_lines.append("### Binary Stage Insights:")
    report_lines.append(f"- Achieved {bin_metrics['accuracy']*100:.2f}% accuracy in distinguishing tumor from no tumor")
    report_lines.append(f"- ROC AUC of {bin_metrics['roc_auc']:.4f} indicates excellent discriminative ability")
    report_lines.append(f"- Tumor detection recall: {bin_metrics['recall'][1]:.4f} (sensitivity)")
    report_lines.append(f"- Tumor detection precision: {bin_metrics['precision'][1]:.4f} (positive predictive value)")
    report_lines.append("")
    
    report_lines.append("### Multi-Class Stage Insights:")
    report_lines.append(f"- Achieved {mc_metrics['accuracy']*100:.2f}% accuracy in classifying tumor types")
    
    # Find best and worst performing classes
    f1_scores = mc_metrics['f1_score']
    best_class_idx = np.argmax(f1_scores)
    worst_class_idx = np.argmin(f1_scores)
    
    report_lines.append(f"- Best performing class: {TUMOR_CLASSES[best_class_idx]} (F1={f1_scores[best_class_idx]:.4f})")
    report_lines.append(f"- Most challenging class: {TUMOR_CLASSES[worst_class_idx]} (F1={f1_scores[worst_class_idx]:.4f})")
    
    # Check balance
    f1_std = np.std(f1_scores)
    if f1_std < 0.05:
        report_lines.append(f"- Balanced performance across all tumor types (F1 std: {f1_std:.4f})")
    else:
        report_lines.append(f"- Some variation in performance across tumor types (F1 std: {f1_std:.4f})")
    
    report_lines.append("")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## CLINICAL DEPLOYMENT RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append("1. **Binary Stage**: High sensitivity ensures minimal missed tumors")
    report_lines.append("2. **Multi-Class Stage**: Provides reliable tumor type classification")
    report_lines.append("3. **Uncertainty Flagging**: Use entropy-based confidence scores to flag uncertain cases")
    report_lines.append("4. **Human Review**: Recommend radiologist review for low-confidence predictions")
    report_lines.append("5. **Pipeline Reliability**: Two-stage approach allows for targeted analysis")
    report_lines.append("")
    report_lines.append("")
    
    # Limitations
    report_lines.append("## LIMITATIONS AND FUTURE IMPROVEMENTS")
    report_lines.append("-" * 80)
    report_lines.append("1. Limited to three tumor types (glioma, meningioma, pituitary)")
    report_lines.append("2. Performance on rare tumor types not evaluated")
    report_lines.append("3. No external validation on different scanner/protocol")
    report_lines.append("4. Consider ensemble methods for improved robustness")
    report_lines.append("5. Explainability methods (Grad-CAM) could enhance clinical trust")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open(save_dir / 'evaluation_report.txt', 'w') as f:
        f.write(report_text)
    
    with open(save_dir / 'evaluation_report.md', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Report saved to:")
    print(f"   - {save_dir / 'evaluation_report.txt'}")
    print(f"   - {save_dir / 'evaluation_report.md'}")
    
    return report_text


def save_results_tables(binary_results, mc_results, save_dir):
    """Save results as CSV tables."""
    print("\n" + "=" * 80)
    print("💾 Saving Results Tables")
    print("=" * 80)
    
    save_dir = Path(save_dir)
    
    # Binary results table
    binary_df = pd.DataFrame({
        'file_path': binary_results['file_paths'],
        'true_class': binary_results['true_labels'],
        'predicted_class': ['notumor' if p == 0 else 'tumor' for p in binary_results['y_pred']],
        'confidence': binary_results['confidence_scores'],
        'tumor_probability': binary_results['y_pred_proba_tumor'],
        'correct': binary_results['y_true'] == binary_results['y_pred']
    })
    binary_df.to_csv(save_dir / 'binary_predictions.csv', index=False)
    print(f"  ✅ Saved: binary_predictions.csv ({len(binary_df)} rows)")
    
    # Multi-class results table
    mc_df = pd.DataFrame({
        'file_path': mc_results['file_paths'],
        'true_class': mc_results['true_labels'],
        'predicted_class': [TUMOR_CLASSES[p] for p in mc_results['y_pred']],
        'confidence': mc_results['confidence_scores'],
        'glioma_prob': mc_results['y_pred_proba'][:, 0],
        'meningioma_prob': mc_results['y_pred_proba'][:, 1],
        'pituitary_prob': mc_results['y_pred_proba'][:, 2],
        'correct': mc_results['y_true'] == mc_results['y_pred']
    })
    mc_df.to_csv(save_dir / 'multiclass_predictions.csv', index=False)
    print(f"  ✅ Saved: multiclass_predictions.csv ({len(mc_df)} rows)")
    
    print(f"\n✅ All tables saved to: {save_dir}")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Load test dataset
    images, labels, file_paths = load_test_dataset()
    
    # Phase 1: Binary Classification
    binary_results = evaluate_binary_model(images, labels, file_paths)
    binary_results = calculate_binary_metrics(binary_results)
    binary_results = calculate_uncertainty(binary_results, stage='binary')
    binary_results = visualize_binary_results(binary_results, RESULTS_DIR / 'binary')
    
    # Phase 2: Multi-Class Classification
    tumor_images, tumor_labels, tumor_paths = filter_for_multiclass(images, labels, file_paths)
    mc_results = evaluate_multiclass_model(tumor_images, tumor_labels, tumor_paths)
    mc_results = calculate_multiclass_metrics(mc_results)
    mc_results = calculate_uncertainty(mc_results, stage='multiclass')
    mc_results = visualize_multiclass_results(mc_results, RESULTS_DIR / 'multiclass')
    
    # Error Analysis
    error_data = perform_error_analysis(binary_results, mc_results, RESULTS_DIR)
    
    # Generate Reports
    save_results_tables(binary_results, mc_results, RESULTS_DIR)
    report = generate_comprehensive_report(binary_results, mc_results, RESULTS_DIR)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All results saved to: {RESULTS_DIR}")
    print(f"\n📊 Summary:")
    print(f"  Binary Accuracy: {binary_results['metrics']['accuracy']*100:.2f}%")
    print(f"  Multi-Class Accuracy: {mc_results['metrics']['accuracy']*100:.2f}%")
    print(f"\n🎉 Evaluation pipeline completed successfully!")
