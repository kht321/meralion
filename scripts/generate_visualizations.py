#!/usr/bin/env python3
"""
Generate visualization charts for robustness evaluation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data():
    """Load robustness summary data."""
    summary_df = pd.read_csv('results/robustness/summary.csv')
    return summary_df

def create_corruption_heatmap(df, output_path):
    """Create heatmap showing ΔWER for each model and corruption."""
    # Pivot data for heatmap
    pivot_data = df[df['corruption'] != 'none'].pivot_table(
        values='dWER_mean',
        index='corruption',
        columns='model',
        aggfunc='max'  # Take worst severity for each corruption type
    )

    # Reorder columns for consistent display
    column_order = ['meralion-2-10b', 'meralion-2-3b', 'whisper-small']
    pivot_data = pivot_data[[col for col in column_order if col in pivot_data.columns]]

    # Rename for display
    pivot_data.columns = ['MERaLiON-2-10B', 'MERaLiON-2-3B', 'Whisper-small']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_data * 100,  # Convert to percentage points
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'ΔWER (pp)'},
        ax=ax,
        vmin=0,
        vmax=80
    )

    ax.set_title('Worst-Case ΔWER by Corruption Type and Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Corruption Type', fontsize=12)

    # Improve corruption labels
    corruption_labels = {
        'clipping_ratio': 'Clipping',
        'noise_snr_db': 'Noise (SNR)',
        'pitch_semitones': 'Pitch Shift',
        'reverb_decay': 'Reverberation',
        'speed': 'Speed Change'
    }
    ax.set_yticklabels([corruption_labels.get(label.get_text(), label.get_text())
                        for label in ax.get_yticklabels()])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created heatmap: {output_path}")

def create_clean_vs_robust_plot(df, output_path):
    """Create scatter plot: clean WER vs worst ΔWER."""
    # Get clean WER and worst degradation for each model
    clean_wer = df[df['corruption'] == 'none'][['model', 'wer_mean']].copy()
    worst_degradation = df[df['corruption'] != 'none'].groupby('model')['dWER_mean'].max().reset_index()

    plot_data = clean_wer.merge(worst_degradation, on='model')
    plot_data.columns = ['model', 'clean_wer', 'worst_dwer']

    # Model display names and colors
    model_info = {
        'meralion-2-10b': ('MERaLiON-2-10B', '#2E86AB'),
        'meralion-2-3b': ('MERaLiON-2-3B', '#A23B72'),
        'whisper-small': ('Whisper-small', '#F18F01')
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, row in plot_data.iterrows():
        model_key = row['model']
        if model_key in model_info:
            name, color = model_info[model_key]
            ax.scatter(
                row['clean_wer'] * 100,
                row['worst_dwer'] * 100,
                s=300,
                color=color,
                alpha=0.7,
                edgecolors='black',
                linewidth=2,
                label=name
            )
            ax.annotate(
                name,
                (row['clean_wer'] * 100, row['worst_dwer'] * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )

    ax.set_xlabel('Clean WER (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Worst ΔWER (pp)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Robustness Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add quadrant labels
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created scatter plot: {output_path}")

def create_corruption_severity_plot(df, output_path):
    """Create line plot showing how ΔWER changes with severity."""
    # Focus on noise corruption (most interesting)
    noise_df = df[df['corruption'] == 'noise_snr_db'].copy()

    model_info = {
        'meralion-2-10b': ('MERaLiON-2-10B', '#2E86AB', 'o'),
        'meralion-2-3b': ('MERaLiON-2-3B', '#A23B72', 's'),
        'whisper-small': ('Whisper-small', '#F18F01', '^')
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_key, (name, color, marker) in model_info.items():
        model_data = noise_df[noise_df['model'] == model_key].sort_values('severity')
        ax.plot(
            model_data['severity'],
            model_data['dWER_mean'] * 100,
            label=name,
            color=color,
            marker=marker,
            markersize=10,
            linewidth=2.5,
            alpha=0.8
        )

    ax.set_xlabel('Noise SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔWER (pp)', fontsize=12, fontweight='bold')
    ax.set_title('Noise Robustness: ΔWER vs Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Lower SNR (noisier) on the right

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created line plot: {output_path}")

def create_model_comparison_bar(df, output_path):
    """Create bar chart comparing models across key metrics."""
    # Get clean WER and average/worst degradation
    clean_wer = df[df['corruption'] == 'none'][['model', 'wer_mean', 'cer_mean']].copy()

    # Average and worst degradation (excluding clean)
    corrupted = df[df['corruption'] != 'none'].groupby('model').agg({
        'dWER_mean': ['mean', 'max']
    }).reset_index()
    corrupted.columns = ['model', 'avg_dwer', 'worst_dwer']

    plot_data = clean_wer.merge(corrupted, on='model')

    # Rename models
    model_names = {
        'meralion-2-10b': 'MERaLiON\n2-10B',
        'meralion-2-3b': 'MERaLiON\n2-3B',
        'whisper-small': 'Whisper\nsmall'
    }
    plot_data['model_display'] = plot_data['model'].map(model_names)

    # Create grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # Clean WER
    axes[0].bar(plot_data['model_display'], plot_data['wer_mean'] * 100, color=colors, alpha=0.8)
    axes[0].set_ylabel('WER (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Clean Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 35)
    axes[0].grid(axis='y', alpha=0.3)

    # Average ΔWER
    axes[1].bar(plot_data['model_display'], plot_data['avg_dwer'] * 100, color=colors, alpha=0.8)
    axes[1].set_ylabel('Avg ΔWER (pp)', fontsize=11, fontweight='bold')
    axes[1].set_title('Average Robustness', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 12)
    axes[1].grid(axis='y', alpha=0.3)

    # Worst ΔWER
    axes[2].bar(plot_data['model_display'], plot_data['worst_dwer'] * 100, color=colors, alpha=0.8)
    axes[2].set_ylabel('Worst ΔWER (pp)', fontsize=11, fontweight='bold')
    axes[2].set_title('Worst-Case Robustness', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 85)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created bar chart: {output_path}")

def main():
    """Generate all visualization charts."""
    output_dir = Path('results/robustness/charts')
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading robustness data...")
    df = load_data()

    print("\nGenerating visualizations...")

    # 1. Heatmap of worst ΔWER by corruption type
    create_corruption_heatmap(df, output_dir / 'corruption_heatmap.png')

    # 2. Accuracy vs robustness scatter plot
    create_clean_vs_robust_plot(df, output_dir / 'accuracy_vs_robustness.png')

    # 3. Noise severity line plot
    create_corruption_severity_plot(df, output_dir / 'noise_severity.png')

    # 4. Model comparison bar chart
    create_model_comparison_bar(df, output_dir / 'model_comparison.png')

    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("\nGenerated charts:")
    print("  1. corruption_heatmap.png - Heatmap of ΔWER by corruption type")
    print("  2. accuracy_vs_robustness.png - Scatter plot of clean WER vs worst ΔWER")
    print("  3. noise_severity.png - Line plot of noise robustness across SNR levels")
    print("  4. model_comparison.png - Bar charts comparing key metrics")

if __name__ == '__main__':
    main()
