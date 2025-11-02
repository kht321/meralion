#!/usr/bin/env python3
"""
Visualize robustness metrics for Self-Curated Conversational Dataset.
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

# Data from README.md table (Self-Curated Conversational Dataset)
data = {
    'Model': ['MERaLiON-2-10B', 'MERaLiON-2-3B', 'Whisper-small'],
    'Clean WER': [66.5, 38.8, 52.2],
    'Clean CER': [53.0, 29.7, 42.8],
    'Avg ΔWER': [-8.5, 2.8, 6.4],
    'Worst ΔWER': [0.4, 28.2, 38.5]
}

# Create output directory
output_dir = Path('results/self_curated/charts')
output_dir.mkdir(parents=True, exist_ok=True)

# Color scheme
colors = {
    'MERaLiON-2-10B': '#e74c3c',
    'MERaLiON-2-3B': '#3498db',
    'Whisper-small': '#95a5a6'
}

# ============================================================================
# Figure 1: Clean WER Comparison (Main Focus)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(data['Model']))
bars = ax.bar(x, data['Clean WER'], color=[colors[m] for m in data['Model']],
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels on bars
for i, (bar, wer) in enumerate(zip(bars, data['Clean WER'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{wer}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Clean WER on Self-Curated Conversational Dataset\n(20 Singlish Audio Samples)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(data['Model'], fontsize=11)
ax.set_ylim(0, max(data['Clean WER']) * 1.15)

# Add grid
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add annotation for best performer
best_idx = data['Clean WER'].index(min(data['Clean WER']))
ax.annotate('Best: 27.7pp better\nthan 10B variant',
            xy=(best_idx, data['Clean WER'][best_idx]),
            xytext=(best_idx + 0.5, data['Clean WER'][best_idx] + 10),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / 'clean_wer_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'clean_wer_comparison.png'}")
plt.close()

# ============================================================================
# Figure 2: Clean WER vs Clean CER Scatter
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

for i, model in enumerate(data['Model']):
    ax.scatter(data['Clean WER'][i], data['Clean CER'][i],
               s=500, color=colors[model], edgecolor='black',
               linewidth=2, alpha=0.85, label=model, zorder=3)

    # Add model name next to point
    ax.annotate(model,
                xy=(data['Clean WER'][i], data['Clean CER'][i]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[model],
                         alpha=0.3, edgecolor='black'))

ax.set_xlabel('Clean WER (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Clean CER (%)', fontsize=12, fontweight='bold')
ax.set_title('Clean WER vs CER on Conversational Singlish\n(Lower is Better)',
             fontsize=14, fontweight='bold', pad=20)

# Add diagonal reference line
max_val = max(max(data['Clean WER']), max(data['Clean CER']))
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='WER = CER')

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.legend(loc='upper left', fontsize=10)

# Add quadrant labels
ax.text(30, 50, 'High WER\nHigh CER', ha='center', va='center',
        fontsize=9, alpha=0.4, style='italic')

plt.tight_layout()
plt.savefig(output_dir / 'clean_wer_vs_cer.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'clean_wer_vs_cer.png'}")
plt.close()

# ============================================================================
# Figure 3: Robustness Overview (4 Metrics)
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
        bar_colors = ['green' if v < 0 else 'red' for v in values]
    else:
        bar_colors = [colors[m] for m in data['Model']]

    bars = ax.bar(x, values, color=bar_colors, edgecolor='black',
                   linewidth=1.5, alpha=0.75)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label_y = height + (max(values) - min(values)) * 0.03
        if 'Δ' in metric and val < 0:
            label_y = height - (max(values) - min(values)) * 0.05
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{val:+.1f}pp' if 'Δ' in metric else f'{val}%',
                ha='center', va='bottom' if height >= 0 else 'top',
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

plt.suptitle('Robustness Metrics: Self-Curated Conversational Dataset',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'robustness_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'robustness_overview.png'}")
plt.close()

# ============================================================================
# Figure 3B: Enhanced Robustness Summary (Presentation-Ready)
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
            if v < 0:
                bar_colors.append('#2ecc71')  # Green for improvement
            elif v < 5:
                bar_colors.append('#f39c12')  # Orange for moderate degradation
            else:
                bar_colors.append('#e74c3c')  # Red for severe degradation
    else:
        # For Clean WER, use performance-based colors
        bar_colors = []
        for v in values:
            if v < 40:
                bar_colors.append('#2ecc71')  # Green for good performance
            elif v < 55:
                bar_colors.append('#f39c12')  # Orange for moderate
            else:
                bar_colors.append('#e74c3c')  # Red for poor performance

    bars = ax.bar(x, values, color=bar_colors, edgecolor='black',
                   linewidth=2, alpha=0.85, width=0.6)

    # Enhanced value labels with backgrounds
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()

        # Position label
        if is_delta and val < 0:
            label_y = height - abs(max(values) - min(values)) * 0.08
            va = 'top'
        else:
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
    if metric == 'Clean WER':
        best_idx = values.index(min(values))
    elif metric == 'Avg ΔWER':
        best_idx = values.index(min(values))
    else:  # Worst ΔWER
        best_idx = values.index(min(values))

    # Add star annotation for best performer
    ax.text(best_idx, ax.get_ylim()[1] * 0.95, '★ BEST',
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
        # Highlight the gap between best and worst
        best_val = min(values)
        worst_val = max(values)
        gap = worst_val - best_val
        ax.text(1, worst_val * 0.7, f'{gap:.1f}pp gap\nbetween best\n& worst',
                ha='center', va='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                         edgecolor='orange', linewidth=2, alpha=0.8))

plt.suptitle('Key Robustness Metrics: Self-Curated Conversational Dataset',
             fontsize=18, fontweight='bold', y=1.02)

# Add footer note
fig.text(0.5, -0.02, 'Evaluated on 20 Singlish conversational samples with multi-speaker code-switching',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig(output_dir / 'robustness_summary_presentation.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'robustness_summary_presentation.png'}")
plt.close()

# ============================================================================
# Figure 4: Clean WER with Domain Shift Context (NSC vs Conversational)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# NSC Part 1 baseline data from README
nsc_data = {
    'MERaLiON-2-10B': 13.6,
    'MERaLiON-2-3B': 13.1,
    'Whisper-small': 17.9
}

x = np.arange(len(data['Model']))
width = 0.35

bars1 = ax.bar(x - width/2, [nsc_data[m] for m in data['Model']],
               width, label='NSC Part 1 (Clean Read Speech)',
               color='lightgreen', edgecolor='black', linewidth=1.5, alpha=0.8)

bars2 = ax.bar(x + width/2, data['Clean WER'],
               width, label='Self-Curated (Conversational Singlish)',
               color=[colors[m] for m in data['Model']],
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add domain shift arrows
for i, model in enumerate(data['Model']):
    nsc_wer = nsc_data[model]
    conv_wer = data['Clean WER'][i]
    shift = conv_wer - nsc_wer

    # Arrow from NSC to Conversational
    ax.annotate('', xy=(i + width/2, conv_wer - 2),
                xytext=(i - width/2, nsc_wer + 2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.6))

    # Domain shift label
    mid_y = (nsc_wer + conv_wer) / 2
    ax.text(i, mid_y, f'+{shift:.1f}pp', ha='center', va='center',
            fontsize=9, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Domain Shift Impact: NSC Read Speech vs Conversational Singlish\n(Red arrows show WER degradation)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(data['Model'], fontsize=11)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(output_dir / 'domain_shift_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'domain_shift_comparison.png'}")
plt.close()

print("\n✅ All visualizations generated successfully!")
print(f"Output directory: {output_dir.absolute()}")
