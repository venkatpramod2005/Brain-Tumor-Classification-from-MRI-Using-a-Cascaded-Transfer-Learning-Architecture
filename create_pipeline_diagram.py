"""
Generate Professional Pipeline Methodology Diagram
Creates a beautiful, colorful flowchart showing the complete system architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure with high DPI for publication quality
fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme - professional and beautiful
colors = {
    'data': '#4A90E2',           # Blue
    'training': '#7B68EE',       # Medium slate blue
    'model': '#50C878',          # Emerald green
    'evaluation': '#FF6B6B',     # Coral red
    'deployment': '#FFA500',     # Orange
    'uncertainty': '#9B59B6',    # Purple
    'box_edge': '#2C3E50',       # Dark blue-gray
    'arrow': '#34495E',          # Dark gray
    'text': '#2C3E50'            # Dark text
}

def create_box(ax, x, y, width, height, text, color, alpha=0.9, fontsize=11, bold=False):
    """Create a rounded rectangle box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1", 
        edgecolor=colors['box_edge'],
        facecolor=color,
        alpha=alpha,
        linewidth=2.5
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color='white', weight=weight,
            wrap=True)
    return box

def create_arrow(ax, x1, y1, x2, y2, style='->',  linewidth=2.5):
    """Create an arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=colors['arrow'],
        linewidth=linewidth,
        mutation_scale=25,
        zorder=1
    )
    ax.add_patch(arrow)
    return arrow

# Title
ax.text(8, 11.3, 'Brain Tumor Intelligence System - Pipeline Methodology', 
        ha='center', va='top', fontsize=20, weight='bold', color=colors['text'])

# ============= STAGE 1: DATA PREPARATION =============
ax.text(1.5, 10.5, 'Data Preparation', ha='left', va='top', 
        fontsize=14, weight='bold', color=colors['data'])

# Training Data
create_box(ax, 2, 9.5, 2.2, 0.8, 'Training Data\n5,712 MRI Images', colors['data'], fontsize=10, bold=True)

# Testing Data
create_box(ax, 2, 8.3, 2.2, 0.8, 'Testing Data\n1,311 MRI Images', colors['data'], fontsize=10, bold=True)

# Data split details
create_box(ax, 2, 7.1, 2.2, 1.0, 'Classes:\n• Glioma\n• Meningioma\n• Pituitary\n• No Tumor', 
           colors['data'], alpha=0.7, fontsize=9)

# ============= STAGE 2: MODEL TRAINING =============
ax.text(5.5, 10.5, 'Model Training & Optimization', ha='left', va='top', 
        fontsize=14, weight='bold', color=colors['training'])

# Training Pipeline Box (Large)
training_box_x, training_box_y = 7, 8
create_box(ax, training_box_x, training_box_y, 3.5, 3.8, '', colors['training'], alpha=0.15, fontsize=10)

# Hyperparameters
create_box(ax, 7, 9.5, 2.8, 0.7, 'Hyperparameters', colors['training'], fontsize=10, bold=True)
create_box(ax, 7, 8.6, 2.8, 0.8, 'ResNet50 Backbone\nDropout: 0.4\nAdam Optimizer', 
           colors['training'], alpha=0.8, fontsize=8.5)

# Training Process
create_box(ax, 7, 7.5, 2.8, 1.0, 'Transfer Learning\nBinary Stage\nMulti-Class Stage', 
           colors['training'], alpha=0.9, fontsize=9)

# Grid Search / Optimization
create_box(ax, 7, 6.3, 2.8, 0.7, 'Optimization\n(Early Stopping)', 
           colors['training'], alpha=0.8, fontsize=9)

# Arrows from data to training
create_arrow(ax, 3.1, 9.5, 5.3, 9.2)  # Training data to hyperparameters
create_arrow(ax, 3.1, 8.3, 5.3, 8.0)  # Testing data to training

# ============= STAGE 3: TRAINED MODELS =============
ax.text(10.5, 10.5, 'Trained Models', ha='left', va='top', 
        fontsize=14, weight='bold', color=colors['model'])

# Best Binary Model
create_box(ax, 11.5, 9.3, 2.5, 0.9, 'Binary Model\n95.80% Accuracy\n(Tumor Detection)', 
           colors['model'], fontsize=10, bold=True)

# Best Multi-Class Model
create_box(ax, 11.5, 8.0, 2.5, 0.9, 'Multi-Class Model\n84.11% Accuracy\n(Tumor Type)', 
           colors['model'], fontsize=10, bold=True)

# MC Dropout Integration
create_box(ax, 11.5, 6.7, 2.5, 0.9, 'MC Dropout Module\n40% Dropout Rate\n20 Forward Passes', 
           colors['uncertainty'], fontsize=9.5, bold=True)

# Arrows from training to models
create_arrow(ax, 8.75, 8.5, 10.2, 9.0)  # Training to binary model
create_arrow(ax, 8.75, 7.8, 10.2, 8.0)  # Training to multi-class model
create_arrow(ax, 11.5, 7.55, 11.5, 7.15)  # Multi-class to MC Dropout

# ============= STAGE 4: EVALUATION =============
ax.text(5.5, 5.2, 'Evaluation Pipeline', ha='left', va='top', 
        fontsize=14, weight='bold', color=colors['evaluation'])

# Evaluation Box
create_box(ax, 7, 4, 2.8, 1.6, 'Comprehensive Evaluation\n\n• Confusion Matrices\n• ROC & PR Curves\n• Error Analysis\n• Uncertainty Estimation', 
           colors['evaluation'], fontsize=9)

# Testing Process
create_box(ax, 11.5, 4, 2.5, 0.8, 'Testing on\n1,311 Test Images', 
           colors['evaluation'], alpha=0.8, fontsize=9.5, bold=True)

# Statistical Analysis
create_box(ax, 7, 2.5, 2.8, 0.7, 'Statistical Validation\nt-tests, ANOVA, Bootstrap', 
           colors['evaluation'], alpha=0.8, fontsize=9)

# Arrows to evaluation
create_arrow(ax, 11.5, 6.2, 11.5, 4.4)  # Models to testing
create_arrow(ax, 10.2, 4.0, 8.4, 4.0)   # Testing to evaluation
create_arrow(ax, 3.1, 7.5, 5.7, 4.5)    # Testing data to evaluation (curved path approximation)

# ============= STAGE 5: RESULTS & DEPLOYMENT =============
ax.text(1, 5.2, 'Results Generation', ha='left', va='top', 
        fontsize=14, weight='bold', color=colors['deployment'])

# Results Box
create_box(ax, 2, 3.5, 2.2, 1.8, 'Publication-Ready\nOutputs\n\n• 6 Visualizations\n• CSV Reports\n• Markdown Docs\n• JSON Analysis', 
           colors['deployment'], fontsize=8.5)

# Arrow from evaluation to results
create_arrow(ax, 5.6, 3.7, 3.1, 3.5)

# ============= STAGE 6: DEPLOYMENT =============
ax.text(10.5, 2.5, 'Production Deployment', ha='left', va='top', 
        fontsize=14, weight='bold', color=colors['deployment'])

# Web Application
create_box(ax, 11.5, 1.5, 2.5, 1.2, 'Streamlit Web App\n\n• Real-time Inference\n• Interactive UI\n• Confidence Gauges\n• MC Dropout Toggle', 
           colors['deployment'], fontsize=9)

# Cloud Deployment
create_box(ax, 14.5, 1.5, 2.2, 0.8, 'Cloud Deployment\n(Streamlit Cloud)', 
           colors['deployment'], alpha=0.8, fontsize=9, bold=True)

# Arrows to deployment
create_arrow(ax, 11.5, 3.6, 11.5, 2.1)  # Evaluation to web app
create_arrow(ax, 12.75, 1.5, 13.4, 1.5)  # Web app to cloud

# ============= FEEDBACK LOOPS =============
# Feedback from evaluation to training (improvement loop)
create_arrow(ax, 5.6, 4.5, 5.3, 7.0, style='<-', linewidth=2)
ax.text(4.5, 5.7, 'Model\nRefinement', ha='center', va='center',
        fontsize=8, style='italic', color=colors['arrow'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=colors['arrow']))

# ============= KEY FEATURES BOXES =============
# Bottom left: Key Metrics
metrics_box_x, metrics_box_y = 2, 0.8
create_box(ax, metrics_box_x, metrics_box_y, 3, 0.9, 
           'Key Performance Metrics\nBinary: 95.80% Acc, 0.9882 ROC-AUC | Multi-Class: 84.11% Acc | Sensitivity: 96.36%',
           colors['model'], alpha=0.85, fontsize=8)

# Bottom right: Novel Features
features_box_x, features_box_y = 7, 0.8
create_box(ax, features_box_x, features_box_y, 4, 0.9,
           'Novel Contributions: Dual-Stage Architecture • MC Dropout Uncertainty • Clinical Review Flagging • Real-time Deployment',
           colors['uncertainty'], alpha=0.85, fontsize=8)

# Add legend for stages
legend_elements = [
    mpatches.Patch(facecolor=colors['data'], edgecolor=colors['box_edge'], label='Data'),
    mpatches.Patch(facecolor=colors['training'], edgecolor=colors['box_edge'], label='Training'),
    mpatches.Patch(facecolor=colors['model'], edgecolor=colors['box_edge'], label='Models'),
    mpatches.Patch(facecolor=colors['uncertainty'], edgecolor=colors['box_edge'], label='Uncertainty'),
    mpatches.Patch(facecolor=colors['evaluation'], edgecolor=colors['box_edge'], label='Evaluation'),
    mpatches.Patch(facecolor=colors['deployment'], edgecolor=colors['box_edge'], label='Deployment')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
          framealpha=0.95, edgecolor=colors['box_edge'], 
          bbox_to_anchor=(0.99, 0.99))

# Add watermark/citation
ax.text(8, 0.15, 'Clinically-Aware Multi-Stage Brain Tumor Intelligence System | 2026',
        ha='center', va='bottom', fontsize=8, style='italic', color=colors['text'], alpha=0.6)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = r"C:\Users\venkat\Downloads\Clinically-Aware Multi-Stage Brain Tumor Intelligence System\visualizations\pipeline_methodology_diagram.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Pipeline methodology diagram saved to: {output_path}")
print(f"   • Resolution: 300 DPI (publication quality)")
print(f"   • Format: PNG with RGB colors")
print(f"   • Size: 16×12 inches (suitable for papers/presentations)")

# Also create a smaller version for web use
output_path_web = output_path.replace('.png', '_web.png')
plt.savefig(output_path_web, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Web version saved to: {output_path_web}")

plt.show()
print("\n📊 Pipeline methodology diagram generation complete!")
