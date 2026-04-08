"""
Generate Statistical Significance Visualizations
Creates publication-quality plots showing statistical validation results
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = Path(__file__).parent
STATS_DIR = BASE_DIR / "documentation"
OUTPUT_DIR = BASE_DIR / "visualizations"

print("=" * 80)
print("Statistical Significance Visualization Generator")
print("=" * 80)

# Load statistical analysis report
with open(STATS_DIR / "statistical_analysis_report.json", 'r') as f:
    stats_data = json.load(f)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10), dpi=300)

# ============= SUBPLOT 1: Confidence Intervals =============
ax1 = plt.subplot(2, 3, 1)

binary_acc = stats_data['analysis_summary']['binary_classification']['accuracy']
binary_ci = stats_data['analysis_summary']['binary_classification']['confidence_interval_95']
mc_acc = stats_data['analysis_summary']['multiclass_classification']['overall_accuracy']
mc_ci = stats_data['analysis_summary']['multiclass_classification']['confidence_interval_95']

categories = ['Binary\nClassification', 'Multi-Class\nClassification']
accuracies = [binary_acc, mc_acc]
ci_lowers = [binary_ci['lower'], mc_ci['lower']]
ci_uppers = [binary_ci['upper'], mc_ci['upper']]
errors_lower = [acc - ci_low for acc, ci_low in zip(accuracies, ci_lowers)]
errors_upper = [ci_up - acc for acc, ci_up in zip(accuracies, ci_uppers)]

x_pos = np.arange(len(categories))
bars = ax1.bar(x_pos, accuracies, color=['#4A90E2', '#50C878'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.errorbar(x_pos, accuracies, yerr=[errors_lower, errors_upper], fmt='none', 
             color='black', capsize=8, capthick=2, linewidth=2)

# Add accuracy labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy with 95% Confidence Intervals\n(Bootstrap, n=1000)', 
              fontsize=12, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(categories, fontsize=11)
ax1.set_ylim(0.75, 1.0)
ax1.grid(axis='y', alpha=0.3)

# ============= SUBPLOT 2: Per-Class Performance =============
ax2 = plt.subplot(2, 3, 2)

per_class = stats_data['analysis_summary']['multiclass_classification']['per_class_accuracy']
classes = list(per_class.keys())
class_accs = [per_class[cls] for cls in classes]

colors_class = ['#7B68EE', '#FF6B6B', '#FFA500']
bars2 = ax2.bar(range(len(classes)), class_accs, color=colors_class, alpha=0.8, 
                edgecolor='black', linewidth=1.5)

# Add significance markers based on ANOVA
ax2.text(0.5, 0.95, '***', ha='center', va='top', fontsize=16, fontweight='bold',
         transform=ax2.transAxes)
ax2.text(0.5, 0.92, 'p < 0.001', ha='center', va='top', fontsize=9, style='italic',
         transform=ax2.transAxes)

for bar, acc, cls in zip(bars2, class_accs, classes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Per-Class Accuracy (Multi-Class)\nANOVA: F=46.98, p<0.001***', 
              fontsize=12, fontweight='bold', pad=15)
ax2.set_xticks(range(len(classes)))
ax2.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', alpha=0.3)

# ============= SUBPLOT 3: Confidence Stratification =============
ax3 = plt.subplot(2, 3, 3)

mc_strat = stats_data['analysis_summary']['confidence_validation']['multiclass']['stratified_accuracy']
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
strat_accs = [mc_strat[f'confidence_ge_{t}']['accuracy'] for t in thresholds]
strat_pcts = [mc_strat[f'confidence_ge_{t}']['percentage_of_cases'] for t in thresholds]

line = ax3.plot(thresholds, strat_accs, marker='o', linewidth=3, markersize=10, 
                color='#9B59B6', label='Accuracy')
ax3_twin = ax3.twinx()
bars3 = ax3_twin.bar(thresholds, strat_pcts, alpha=0.3, color='#34495E', width=0.03,
                     label='% Cases')

ax3.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color='#9B59B6')
ax3_twin.set_ylabel('Percentage of Cases', fontsize=12, fontweight='bold', color='#34495E')
ax3.set_title('Confidence Calibration\n(Higher Confidence → Higher Accuracy)', 
              fontsize=12, fontweight='bold', pad=15)
ax3.tick_params(axis='y', labelcolor='#9B59B6')
ax3_twin.tick_params(axis='y', labelcolor='#34495E')
ax3.grid(alpha=0.3)
ax3.set_ylim(0.80, 1.0)

# Add correlation annotation
corr = stats_data['analysis_summary']['confidence_validation']['binary']['confidence_correctness_correlation']
ax3.text(0.05, 0.05, f"Pearson r = {corr['pearson_r']:.4f}\np < 0.001***", 
         transform=ax3.transAxes, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============= SUBPLOT 4: Sensitivity vs Specificity =============
ax4 = plt.subplot(2, 3, 4)

binary_metrics = stats_data['analysis_summary']['binary_classification']
sensitivity = binary_metrics['sensitivity']
specificity = binary_metrics['specificity']

metrics_vals = [sensitivity, specificity]
metrics_names = ['Sensitivity\n(TPR)', 'Specificity\n(TNR)']
colors_metrics = ['#1f8f4d', '#0d3b0d']

bars4 = ax4.bar(range(2), metrics_vals, color=colors_metrics, alpha=0.8,
                edgecolor='black', linewidth=1.5)

for bar, val, name in zip(bars4, metrics_vals, metrics_names):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{val*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax4.axhline(y=0.95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='95% Threshold')
ax4.set_ylabel('Rate', fontsize=12, fontweight='bold')
ax4.set_title('Binary Classification: Sensitivity vs Specificity\nBalanced Performance (Δ=1.79%)', 
              fontsize=12, fontweight='bold', pad=15)
ax4.set_xticks(range(2))
ax4.set_xticklabels(metrics_names, fontsize=11)
ax4.set_ylim(0.90, 1.0)
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# ============= SUBPLOT 5: Statistical Significance Summary =============
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

# Create text summary
summary_text = """Statistical Validation Summary

Binary vs Random Baseline:
  • t-statistic: 32.21
  • p-value: < 0.001 ***
  ✓ Highly Significant

Multi-Class ANOVA:
  • F-statistic: 46.98
  • p-value: < 0.001 ***
  ✓ Significant Class Differences

Confidence Calibration:
  • Pearson r: 0.3894
  • p-value: < 0.001 ***
  ✓ Well-Calibrated

Bootstrap Validation:
  • Iterations: 1000
  • Method: Percentile
  ✓ Robust Confidence Intervals

Significance Level: α = 0.05
All tests: p < 0.001 ***
"""

ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

# ============= SUBPLOT 6: Confusion Pattern Heatmap =============
ax6 = plt.subplot(2, 3, 6)

error_pairs = stats_data['analysis_summary']['error_patterns']['most_confused_pairs']
pair_labels = [f"{p['class_1'].capitalize()}\n↔\n{p['class_2'].capitalize()}" for p in error_pairs]
pair_counts = [p['confusion_count'] for p in error_pairs]

bars6 = ax6.barh(range(len(pair_labels)), pair_counts, color='#e06f6f', alpha=0.8,
                 edgecolor='black', linewidth=1.5)

for bar, count in zip(bars6, pair_counts):
    width = bar.get_width()
    ax6.text(width + 1, bar.get_y() + bar.get_height()/2.,
             f'{count}', ha='left', va='center', fontweight='bold', fontsize=10)

ax6.set_xlabel('Number of Confusions', fontsize=12, fontweight='bold')
ax6.set_title('Most Confused Class Pairs\n(Multi-Class Classification)', 
              fontsize=12, fontweight='bold', pad=15)
ax6.set_yticks(range(len(pair_labels)))
ax6.set_yticklabels(pair_labels, fontsize=10)
ax6.grid(axis='x', alpha=0.3)

# Overall title
fig.suptitle('Statistical Significance Analysis - Brain Tumor Classification System', 
             fontsize=16, fontweight='bold', y=0.98)

# Add footer
fig.text(0.5, 0.01, 
         'All tests significant at p < 0.001 level | Bootstrap CI: 1000 iterations | Significance: *** p<0.001, ** p<0.01, * p<0.05',
         ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])

# Save
output_path = OUTPUT_DIR / "statistical_significance_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Statistical significance plots saved to: {output_path}")
print(f"   • Resolution: 300 DPI")
print(f"   • Format: 6-panel comprehensive visualization")
print(f"   • Size: 16×10 inches")

plt.show()
print("\n" + "=" * 80)
print("Statistical Visualization Complete!")
print("=" * 80)
