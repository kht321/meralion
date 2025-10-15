"""Generate visualizations for white-box guardrail evaluation results."""

import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def load_jsonl(path):
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]

def create_passthrough_by_category(baseline, intervention, output_path):
    """Create bar chart showing pass-through rate by category."""
    categories = ['benign', 'profanity', 'hate_speech', 'violence', 'pii']
    
    baseline_rates = []
    intervention_rates = []
    
    for cat in categories:
        cat_baseline = [s for s in baseline if s['category'] == cat]
        cat_intervention = [s for s in intervention if s['category'] == cat]
        
        # Count samples with rule hits (harmful keywords detected)
        baseline_hits = sum(1 for s in cat_baseline if s.get('rule_hits', []))
        intervention_hits = sum(1 for s in cat_intervention if s.get('rule_hits', []))
        
        baseline_rates.append(100 * baseline_hits / len(cat_baseline) if cat_baseline else 0)
        intervention_rates.append(100 * intervention_hits / len(cat_intervention) if cat_intervention else 0)
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline (no guardrail)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, intervention_rates, width, label='With logit intervention', color='#3498db')
    
    ax.set_xlabel('Content Category')
    ax.set_ylabel('Harmful Keyword Detection Rate (%)')
    ax.set_title('White-Box Guardrail Evaluation: Harmful Content Pass-Through by Category\nMERaLiON-2-3B')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in categories])
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%' if height > 0 else '0%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created: {output_path}")

def create_latency_comparison(baseline, intervention, output_path):
    """Create box plot comparing latency between baseline and intervention."""
    baseline_latency = [s['latency_ms'] for s in baseline]
    intervention_latency = [s['latency_ms'] for s in intervention]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [baseline_latency, intervention_latency]
    bp = ax.boxplot(data, labels=['Baseline', 'Logit Intervention'], patch_artist=True)
    
    # Color the boxes
    colors = ['#e74c3c', '#3498db']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency: Baseline vs Logit-Level Intervention\nMERaLiON-2-3B (52 samples)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean values as text
    mean_baseline = np.mean(baseline_latency)
    mean_intervention = np.mean(intervention_latency)
    delta = mean_intervention - mean_baseline
    
    ax.text(0.5, 0.95, f'Mean latency delta: +{delta:.1f} ms ({100*delta/mean_baseline:.1f}% overhead)',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created: {output_path}")

def create_trace_exposure_analysis(baseline, intervention, output_path):
    """Create bar chart showing decoder trace exposure (harmful token probabilities)."""
    # Extract trace exposure rates
    baseline_exposure = sum(len([t for t in s.get('decoder_trace', []) if t]) for s in baseline)
    intervention_exposure = sum(len([t for t in s.get('decoder_trace', []) if t]) for s in intervention)
    
    total_baseline_tokens = sum(1 for s in baseline for _ in s.get('decoder_trace', []))
    total_intervention_tokens = sum(1 for s in intervention for _ in s.get('decoder_trace', []))
    
    # For simplicity, show summary stats from the summary file
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Baseline', 'Logit Intervention']
    exposure_rates = [1.10, 1.02]  # From summary file
    
    bars = ax.bar(categories, exposure_rates, color=['#e74c3c', '#3498db'], alpha=0.7)
    
    ax.set_ylabel('Decoder Trace Exposure Rate (%)')
    ax.set_title('Harmful Token Probability in Decoder Trace\nMERaLiON-2-3B')
    ax.set_ylim(0, 2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars, exposure_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{rate:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created: {output_path}")

def create_effectiveness_summary(output_path):
    """Create summary visualization of guardrail effectiveness."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Harmful samples\ntested', 'Keywords detected\n(baseline)', 'Blocked by\nguardrail', 'False blocks\n(benign)', 'Latency\noverhead (ms)']
    values = [40, 9, 0, 0, 31]
    colors = ['#95a5a6', '#e74c3c', '#e67e22', '#27ae60', '#3498db']
    
    bars = ax.barh(metrics, values, color=colors, alpha=0.7)
    
    ax.set_xlabel('Count / Milliseconds')
    ax.set_title('White-Box Guardrail Effectiveness Summary\nMERaLiON-2-3B (Logit-Level Keyword Intervention)')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height()/2., f'  {value}',
               va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created: {output_path}")

def main():
    """Generate all guardrail visualizations."""
    print("Loading guardrail evaluation data...")
    
    baseline_path = Path("results/guardrails/logs/meralion-2-3b/20251012_091317_baseline.jsonl")
    intervention_path = Path("results/guardrails/logs/meralion-2-3b/20251012_091317_intervention.jsonl")
    
    baseline = load_jsonl(baseline_path)
    intervention = load_jsonl(intervention_path)
    
    output_dir = Path("results/guardrails/charts")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nGenerating visualizations...")
    
    create_passthrough_by_category(
        baseline, intervention,
        output_dir / "passthrough_by_category.png"
    )
    
    create_latency_comparison(
        baseline, intervention,
        output_dir / "latency_comparison.png"
    )
    
    create_trace_exposure_analysis(
        baseline, intervention,
        output_dir / "trace_exposure.png"
    )
    
    create_effectiveness_summary(
        output_dir / "effectiveness_summary.png"
    )
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("\nGenerated charts:")
    print("  1. passthrough_by_category.png - Detection rates by content category")
    print("  2. latency_comparison.png - Inference latency overhead")
    print("  3. trace_exposure.png - Harmful token probability in decoder trace")
    print("  4. effectiveness_summary.png - Overall guardrail effectiveness metrics")

if __name__ == "__main__":
    main()
