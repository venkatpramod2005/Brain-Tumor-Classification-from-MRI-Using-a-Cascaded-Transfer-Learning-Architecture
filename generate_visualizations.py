"""
Brain Tumor Classification - Publication-Ready Visualization Module
Generates colorful, attractive visualizations in RGB format
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION ====================

# Paths
BASE_DIR = Path(__file__).parent
VIZ_DIR = BASE_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_DIR = BASE_DIR / "evaluation_results"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
TEST_DIR = DATASET_DIR / "Testing"

# Model paths
BINARY_MODEL_PATH = MODELS_DIR / "best_model_binary_ResNet50_20260331_202827.keras"
MC_MODEL_PATH = MODELS_DIR / "best_model_mc.keras"

# Constants
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3
INPUT_SHAPE = IMG_SIZE + (IMG_CHANNELS,)

# Class definitions
ALL_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
TUMOR_CLASSES = ['glioma', 'meningioma', 'pituitary']

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

# ==================== MATPLOTLIB CONFIGURATION ====================

# Configure matplotlib for colorful, publication-quality output
plt.rcParams['figure.dpi'] = 300  # High resolution
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.figsize'] = (12, 8)

# Colorful color palettes
COLORFUL_PALETTE = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
CLASS_COLORS = {
    'glioma': '#FF6B6B',      # Vibrant red/coral
    'meningioma': '#4ECDC4',  # Turquoise
    'pituitary': '#45B7D1'    # Sky blue
}

print("=" * 80)
print("🎨 Publication-Ready Visualization Module")
print("=" * 80)
print(f"\n📁 Visualization Directory: {VIZ_DIR}")
print(f"🎨 Mode: Colorful RGB Output (300 DPI)")
print(f"✅ Configuration complete!")
print("=" * 80)


# ==================== HELPER FUNCTIONS ====================

def save_figure(fig, filename, dpi=300):
    """Save figure in RGB format with integer pixel values."""
    from PIL import Image
    import io
    
    filepath = VIZ_DIR / filename
    
    # Save to buffer first
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format='png', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    
    # Load image and convert to RGB
    img = Image.open(buf)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save as RGB PNG
    img.save(filepath, 'PNG', dpi=(dpi, dpi))
    
    plt.close(fig)
    print(f"  ✅ Saved: {filename}")
    buf.close()
    return filepath


def load_evaluation_results():
    """Load results from previous evaluation."""
    # Load predictions
    binary_df = pd.read_csv(RESULTS_DIR / 'binary_predictions.csv')
    mc_df = pd.read_csv(RESULTS_DIR / 'multiclass_predictions.csv')
    
    # Load error analysis
    with open(RESULTS_DIR / 'error_analysis.json', 'r') as f:
        error_data = json.load(f)
    
    return binary_df, mc_df, error_data


# ==================== VISUALIZATION 1: CONFUSION MATRIX ====================

def generate_confusion_matrix():
    """Generate colorful multi-class confusion matrix (3×3)."""
    print("\n🎨 Generating Confusion Matrix...")
    
    # Load data
    _, mc_df, _ = load_evaluation_results()
    
    # Get true and predicted labels
    y_true = mc_df['true_class'].map(TUMOR_CLASS_TO_IDX).values
    y_pred = mc_df['predicted_class'].map(TUMOR_CLASS_TO_IDX).values
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with colorful heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use colorful gradient (Blues with high saturation)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=TUMOR_CLASSES,
                yticklabels=TUMOR_CLASSES,
                cbar_kws={'label': 'Number of Samples'},
                ax=ax, square=True, linewidths=2, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_title('Multi-Class Confusion Matrix\n(Tumor Type Classification)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Actual Class', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    save_figure(fig, 'confusion_matrix.png')


# ==================== VISUALIZATION 2: ROC CURVE ====================

def generate_roc_curve():
    """Generate colorful binary ROC curve."""
    print("\n🎨 Generating ROC Curve...")
    
    # Load data
    binary_df, _, _ = load_evaluation_results()
    
    # Get true labels and probabilities
    y_true = (binary_df['true_class'] != 'notumor').astype(int).values
    y_prob = binary_df['tumor_probability'].values
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create colorful figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve with vibrant color
    ax.plot(fpr, tpr, color='#FF6B6B', lw=3, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color='#95A5A6', lw=2, linestyle='--', 
            label='Random Classifier', alpha=0.7)
    
    # Fill area under curve with transparent color
    ax.fill_between(fpr, tpr, alpha=0.3, color='#FF6B6B')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve\n(Binary Tumor Detection)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    save_figure(fig, 'roc_curve.png')


# ==================== VISUALIZATION 3: PRECISION-RECALL CURVE ====================

def generate_precision_recall_curve():
    """Generate colorful precision-recall curve."""
    print("\n🎨 Generating Precision-Recall Curve...")
    
    # Load data
    binary_df, _, _ = load_evaluation_results()
    
    # Get true labels and probabilities
    y_true = (binary_df['true_class'] != 'notumor').astype(int).values
    y_prob = binary_df['tumor_probability'].values
    
    # Compute PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # Create colorful figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot PR curve with vibrant color
    ax.plot(recall, precision, color='#4ECDC4', lw=3, 
            label=f'PR Curve (AUC = {pr_auc:.4f})')
    
    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.3, color='#4ECDC4')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curve\n(Binary Tumor Detection)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="lower left", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    save_figure(fig, 'precision_recall_curve.png')


# ==================== VISUALIZATION 4: CLASS PERFORMANCE ====================

def generate_class_performance():
    """Generate colorful class-wise performance bar chart."""
    print("\n🎨 Generating Class Performance Chart...")
    
    # Load data
    _, mc_df, _ = load_evaluation_results()
    
    # Get true and predicted labels
    y_true = mc_df['true_class'].map(TUMOR_CLASS_TO_IDX).values
    y_pred = mc_df['predicted_class'].map(TUMOR_CLASS_TO_IDX).values
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2]
    )
    
    # Create colorful bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(TUMOR_CLASSES))
    width = 0.25
    
    # Use different vibrant colors for each metric
    bars1 = ax.bar(x - width, precision, width, label='Precision', 
                   color='#FF6B6B', edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, recall, width, label='Recall', 
                   color='#4ECDC4', edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', 
                   color='#45B7D1', edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Tumor Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Class-Wise Performance Metrics\n(Precision, Recall, F1-Score)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(TUMOR_CLASSES, fontsize=12)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # Add background color for visual appeal
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    save_figure(fig, 'class_performance.png')


# ==================== VISUALIZATION 5: CONFIDENCE DISTRIBUTION ====================

def generate_confidence_distribution():
    """Generate colorful confidence distribution plot."""
    print("\n🎨 Generating Confidence Distribution...")
    
    # Load data
    _, mc_df, _ = load_evaluation_results()
    
    # Get confidence scores and true labels
    confidence = mc_df['confidence'].values
    y_true = mc_df['true_class'].values
    
    # Create colorful figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall distribution with gradient color
    n, bins, patches = ax1.hist(confidence, bins=40, edgecolor='white', linewidth=1.2)
    
    # Color gradient from low to high confidence
    cm = plt.cm.get_cmap('RdYlGn')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    ax1.axvline(0.7, color='#E74C3C', linestyle='--', linewidth=2, 
                label='Low Confidence Threshold')
    ax1.set_xlabel('Confidence Score', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax1.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_facecolor('#F8F9FA')
    
    # Per-class distribution with different colors
    for i, (tumor_class, color) in enumerate(CLASS_COLORS.items()):
        class_mask = y_true == tumor_class
        class_conf = confidence[class_mask]
        ax2.hist(class_conf, bins=30, alpha=0.6, label=tumor_class.capitalize(), 
                color=color, edgecolor='white', linewidth=1)
    
    ax2.set_xlabel('Confidence Score', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax2.set_title('Confidence Distribution by Class', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    save_figure(fig, 'confidence_distribution.png')


# ==================== VISUALIZATION 6: ERROR ANALYSIS ====================

def generate_error_analysis():
    """Generate colorful error analysis chart."""
    print("\n🎨 Generating Error Analysis Chart...")
    
    # Load data
    _, mc_df, error_data = load_evaluation_results()
    
    # Get true and predicted labels
    y_true = mc_df['true_class'].map(TUMOR_CLASS_TO_IDX).values
    y_pred = mc_df['predicted_class'].map(TUMOR_CLASS_TO_IDX).values
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate errors per class
    correct_per_class = np.diag(cm)
    total_per_class = cm.sum(axis=1)
    errors_per_class = total_per_class - correct_per_class
    accuracy_per_class = correct_per_class / total_per_class
    
    # Calculate confusion between classes
    confusion_matrix_off_diag = cm.copy()
    np.fill_diagonal(confusion_matrix_off_diag, 0)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Errors per class (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    colors_gradient = ['#2ECC71', '#F39C12', '#E74C3C']
    bars = ax1.bar(TUMOR_CLASSES, errors_per_class, 
                   color=[CLASS_COLORS[c] for c in TUMOR_CLASSES],
                   edgecolor='white', linewidth=2)
    for bar, error in zip(bars, errors_per_class):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(error)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
    ax1.set_title('Misclassifications per Class', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_facecolor('#F8F9FA')
    
    # Subplot 2: Accuracy per class (horizontal bar)
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.barh(TUMOR_CLASSES, accuracy_per_class, 
                    color=[CLASS_COLORS[c] for c in TUMOR_CLASSES],
                    edgecolor='white', linewidth=2)
    for bar, acc in zip(bars, accuracy_per_class):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{acc:.1%}',
                ha='left', va='center', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Classification Accuracy per Class', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1.1])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_facecolor('#F8F9FA')
    
    # Subplot 3: Confusion heatmap (off-diagonal only)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create confusion pairs data
    confusion_data = []
    for i in range(len(TUMOR_CLASSES)):
        for j in range(len(TUMOR_CLASSES)):
            if i != j:
                confusion_data.append({
                    'True': TUMOR_CLASSES[i],
                    'Predicted': TUMOR_CLASSES[j],
                    'Count': cm[i, j]
                })
    
    confusion_df = pd.DataFrame(confusion_data)
    confusion_pivot = confusion_df.pivot(index='True', columns='Predicted', values='Count')
    
    # Heatmap with colorful gradient
    sns.heatmap(confusion_pivot, annot=True, fmt='.0f', cmap='Reds', 
                ax=ax3, cbar_kws={'label': 'Confusion Count'},
                linewidths=2, linecolor='white',
                annot_kws={'size': 13, 'weight': 'bold'})
    ax3.set_title('Class Confusion Matrix (Errors Only)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Class', fontsize=12, fontweight='bold')
    
    # Main title
    fig.suptitle('Comprehensive Error Analysis\n(Multi-Class Tumor Classification)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    save_figure(fig, 'error_analysis.png')


# ==================== VALIDATION FUNCTIONS ====================

def validate_image_format(filepath):
    """Validate that image is in RGB format with integer pixels."""
    from PIL import Image
    
    img = Image.open(filepath)
    
    # Check mode
    if img.mode != 'RGB':
        print(f"  ⚠️  {filepath.name}: Mode is {img.mode}, not RGB")
        return False
    
    # Check data type
    img_array = np.array(img)
    if img_array.dtype != np.uint8:
        print(f"  ⚠️  {filepath.name}: Data type is {img_array.dtype}, not uint8")
        return False
    
    # Check shape
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        print(f"  ⚠️  {filepath.name}: Shape is {img_array.shape}, expected (H, W, 3)")
        return False
    
    # Check value range
    if img_array.min() < 0 or img_array.max() > 255:
        print(f"  ⚠️  {filepath.name}: Values out of range [0, 255]")
        return False
    
    return True


def validate_all_visualizations():
    """Validate all generated visualizations."""
    print("\n" + "=" * 80)
    print("🔍 Validating Visualizations")
    print("=" * 80)
    
    required_files = [
        'confusion_matrix.png',
        'roc_curve.png',
        'precision_recall_curve.png',
        'class_performance.png',
        'confidence_distribution.png',
        'error_analysis.png'
    ]
    
    all_valid = True
    
    for filename in required_files:
        filepath = VIZ_DIR / filename
        
        if not filepath.exists():
            print(f"  ❌ {filename}: File not found")
            all_valid = False
            continue
        
        if validate_image_format(filepath):
            # Get file size
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename}: RGB format, integer pixels, {size_mb:.2f} MB")
        else:
            all_valid = False
    
    return all_valid


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🎨 Generating All Visualizations")
    print("=" * 80)
    
    # Generate all visualizations
    generate_confusion_matrix()
    generate_roc_curve()
    generate_precision_recall_curve()
    generate_class_performance()
    generate_confidence_distribution()
    generate_error_analysis()
    
    # Validate all visualizations
    all_valid = validate_all_visualizations()
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    else:
        print("⚠️  SOME VISUALIZATIONS HAD ISSUES")
    print("=" * 80)
    print(f"\n📁 All visualizations saved to: {VIZ_DIR}")
    print("\n📊 Generated Files:")
    for filename in sorted(VIZ_DIR.glob("*.png")):
        print(f"  • {filename.name}")
    print("\n🎉 Visualization module complete!")

