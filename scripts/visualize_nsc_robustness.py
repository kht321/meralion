#!/usr/bin/env python3
"""
Visualize robustness metrics for NSC Part 1 Dataset (presentation-ready).
Focuses on Clean WER and key robustness metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Data from README.md table (NSC Part 1 robustness sweep)
data = {
    'Model': ['MERaLiON-2-10B', 'MERaLiON-2-3B', 'Whisper-small'],
    'Clean WER': [13.6, 13.1, 17.9],
    'Clean CER': [3.3, 3.1, 6.1],
    'Avg ΔWER': [0.6, 0.5, 9.6],
    'Worst ΔWER': [5.8, 5.1, 77.6]
}

# Create output directory
output_dir = Path('results/robustness/charts')
output_dir.mkdir(parents=True, exist_ok=True)

# Color scheme
colors = {
    'MERaLiON-2-10B': '#e74c3c',
    'MERaLiON-2-3B': '#3498db',
    'Whisper-small': '#95a5a6'
}

# ============================================================================
# Figure 1: Enhanced Robustness Summary (Presentation-Ready)
# ============================================================================
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# Define the three key metrics for presentation
presentation_metrics = [
    ('Clean WER', 'Clean Accuracy\n(Lower is Better)', '%', False),
    ('Avg ΔWER', 'Average Robustness\n(Corruption Impact)', 'pp', True),
    ('Worst ΔWER', 'Worst-Case Robustness\n(Maximum Degradation)', 'pp', True)
]

for idx, (metric, title, unit, is_delta) in enumerate(presentation_metrics):
    ax = fig.add_subplot(gs[0, idx])
    x = np.arange(len(data['Model']))
    values = data[metric]

    # Color coding for better visual hierarchy
    if is_delta:
        bar_colors = []
        for v in values:
            if v < 2:
                bar_colors.append('#2ecc71')  # Green for excellent robustness
            elif v < 10:
                bar_colors.append('#f39c12')  # Orange for moderate degradation
            else:
                bar_colors.append('#e74c3c')  # Red for severe degradation
    else:
        # For Clean WER, use performance-based colors
        bar_colors = []
        for v in values:
            if v < 15:
                bar_colors.append('#2ecc71')  # Green for excellent performance
            elif v < 20:
                bar_colors.append('#f39c12')  # Orange for good
            else:
                bar_colors.append('#e74c3c')  # Red for poor performance

    bars = ax.bar(x, values, color=bar_colors, edgecolor='black',
                   linewidth=2, alpha=0.85, width=0.6)

    # Enhanced value labels with backgrounds
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()

        # Position label
        label_y = height + abs(max(values) - min(values)) * 0.04
        va = 'bottom'

        # Format label
        if is_delta:
            label_text = f'{val:+.1f}{unit}'
        else:
            label_text = f'{val:.1f}{unit}'

        # Add label with background box
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                label_text,
                ha='center', va=va, fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=1.5, alpha=0.9))

    # Highlight best performer with a star
    best_idx = values.index(min(values))

    # Add star annotation for best performer
    ax.text(best_idx, ax.get_ylim()[1] * 0.95, 'BEST',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='gold', bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='black', alpha=0.8))

    # Styling
    ax.set_ylabel(f'Value ({unit})', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(data['Model'], fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    # Add reference line at 0 for delta metrics
    if is_delta:
        ax.axhline(0, color='black', linewidth=2, linestyle='-', alpha=0.6, zorder=1)
        ax.text(len(data['Model']) - 0.5, 0, 'baseline',
                ha='left', va='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Add performance annotations
    if metric == 'Clean WER':
        # Highlight the gap between MERaLiON best and Whisper
        meralion_best = min([data['Clean WER'][i] for i in range(2)])
        whisper_wer = data['Clean WER'][2]
        gap = whisper_wer - meralion_best
        ax.text(1.5, whisper_wer * 0.6, f'{gap:.1f}pp\nMERaLiON\nadvantage',
                ha='center', va='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                         edgecolor='green', linewidth=2, alpha=0.8))

    elif metric == 'Worst ΔWER':
        # Highlight Whisper's catastrophic failure
        whisper_worst = data['Worst ΔWER'][2]
        if whisper_worst > 50:
            ax.text(2, whisper_worst * 0.5, 'Catastrophic\nreverb failure',
                    ha='center', va='center', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffcccc',
                             edgecolor='red', linewidth=2, alpha=0.8))

plt.suptitle('Key Robustness Metrics: NSC Part 1 (682 utterances)',
             fontsize=18, fontweight='bold', y=1.02)

# Add footer note
fig.text(0.5, -0.02, 'Clean read speech with diverse acoustic corruptions (noise, speed, pitch, reverb, clipping)',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig(output_dir / 'robustness_summary_presentation.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'robustness_summary_presentation.png'}")
plt.close()

# ============================================================================
# Figure 2: Clean WER Comparison with Performance Tiers
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(data['Model']))
bars = ax.bar(x, data['Clean WER'], color=[colors[m] for m in data['Model']],
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels on bars
for i, (bar, wer) in enumerate(zip(bars, data['Clean WER'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{wer}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Clean WER on NSC Part 1\n(682 Read Speech Utterances)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(data['Model'], fontsize=11)
ax.set_ylim(0, max(data['Clean WER']) * 1.2)

# Add grid
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add performance tier bands
ax.axhspan(0, 15, alpha=0.1, color='green', zorder=0, label='Excellent (<15%)')
ax.axhspan(15, 20, alpha=0.1, color='orange', zorder=0, label='Good (15-20%)')
ax.axhspan(20, 100, alpha=0.1, color='red', zorder=0, label='Poor (>20%)')

# Add annotation for MERaLiON advantage
meralion_best = min([data['Clean WER'][i] for i in range(2)])
whisper_wer = data['Clean WER'][2]
gap = whisper_wer - meralion_best
ax.annotate(f'{gap:.1f}pp MERaLiON\nadvantage over Whisper',
            xy=(2, whisper_wer),
            xytext=(1.3, whisper_wer + 2),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'clean_wer_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'clean_wer_comparison.png'}")
plt.close()

# ============================================================================
# Figure 3: Robustness Comparison (Avg vs Worst)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

for i, model in enumerate(data['Model']):
    ax.scatter(data['Avg ΔWER'][i], data['Worst ΔWER'][i],
               s=500, color=colors[model], edgecolor='black',
               linewidth=2, alpha=0.85, label=model, zorder=3)

    # Add model name next to point
    offset_x = 1 if model == 'Whisper-small' else -1
    offset_y = 3 if model == 'Whisper-small' else -3
    ax.annotate(model,
                xy=(data['Avg ΔWER'][i], data['Worst ΔWER'][i]),
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[model],
                         alpha=0.3, edgecolor='black'))

ax.set_xlabel('Average ΔWER (pp)', fontsize=12, fontweight='bold')
ax.set_ylabel('Worst-Case ΔWER (pp)', fontsize=12, fontweight='bold')
ax.set_title('Robustness Trade-off: Average vs Worst-Case Degradation\n(Lower-left is better)',
             fontsize=14, fontweight='bold', pad=20)

# Add quadrant shading
ax.axhspan(0, 10, alpha=0.05, color='green', zorder=0)
ax.axvspan(0, 2, alpha=0.05, color='green', zorder=0)

# Add diagonal reference lines
ax.axhline(10, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='Worst ΔWER = 10pp')
ax.axvline(2, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='Avg ΔWER = 2pp')

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add zone labels
ax.text(0.3, 3, 'Excellent\nRobustness', ha='center', va='center',
        fontsize=11, fontweight='bold', alpha=0.4, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

ax.text(50, 60, 'Catastrophic\nFailure Zone', ha='center', va='center',
        fontsize=11, fontweight='bold', alpha=0.6, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffcccc', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'robustness_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'robustness_tradeoff.png'}")
plt.close()

# ============================================================================
# Figure 4: All Metrics Overview (4 panels)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Clean WER', 'Clean CER', 'Avg ΔWER', 'Worst ΔWER']
ylabels = ['WER (%)', 'CER (%)', 'Avg ΔWER (pp)', 'Worst ΔWER (pp)']

for idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
    ax = axes[idx // 2, idx % 2]
    x = np.arange(len(data['Model']))
    values = data[metric]

    # Use diverging colors for delta metrics
    if 'Δ' in metric:
        bar_colors = []
        for v in values:
            if v < 2:
                bar_colors.append('#2ecc71')
            elif v < 10:
                bar_colors.append('#f39c12')
            else:
                bar_colors.append('#e74c3c')
    else:
        bar_colors = [colors[m] for m in data['Model']]

    bars = ax.bar(x, values, color=bar_colors, edgecolor='black',
                   linewidth=1.5, alpha=0.75)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label_y = height + (max(values) - min(values)) * 0.03
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{val:+.1f}pp' if 'Δ' in metric else f'{val}%',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data['Model'], fontsize=9, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add horizontal line at 0 for delta metrics
    if 'Δ' in metric:
        ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

plt.suptitle('Robustness Metrics: NSC Part 1',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'robustness_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'robustness_overview.png'}")
plt.close()

print("\n✅ All NSC Part 1 robustness visualizations generated successfully!")
print(f"Output directory: {output_dir.absolute()}")
