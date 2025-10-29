#!/usr/bin/env python3
"""Recalculate all WER/CER metrics from existing JSONL files with fixed normalization."""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asr_eval.metrics.wer_cer import corpus_scores


def recalculate_robustness():
    """Recalculate robustness evaluation metrics."""
    print("=" * 70)
    print("ROBUSTNESS EVALUATION (NSC Part 1, 682 utterances)")
    print("=" * 70)

    jsonl_path = Path("results/robustness/details.jsonl")
    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        return {}

    # Load all records
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records\n")

    # Group by model and corruption
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        model = rec['model']
        corruption = rec.get('corruption', 'none')
        severity = rec.get('severity', 0)
        key = f"{corruption}_{severity}" if corruption != 'none' else 'clean'
        grouped[model][key].append(rec)

    results = {}

    # Calculate metrics for each model
    for model in sorted(grouped.keys()):
        print(f"\n{model.upper()}")
        print("-" * 70)

        model_results = {}

        # Clean baseline
        if 'clean' in grouped[model]:
            clean_recs = grouped[model]['clean']
            refs = [r['ref'] for r in clean_recs]
            hyps = [r['hyp'] for r in clean_recs]
            scores = corpus_scores(refs, hyps)

            print(f"Clean Baseline:")
            print(f"  WER: {scores.wer*100:5.1f}% | CER: {scores.cer*100:5.1f}%")
            model_results['clean'] = {'wer': scores.wer, 'cer': scores.cer}

        # Corruptions
        corruption_wers = []
        worst_wer = 0
        worst_corruption = ""

        for key in sorted(grouped[model].keys()):
            if key == 'clean':
                continue

            recs = grouped[model][key]
            refs = [r['ref'] for r in recs]
            hyps = [r['hyp'] for r in recs]
            scores = corpus_scores(refs, hyps)

            delta_wer = scores.wer - model_results['clean']['wer']
            corruption_wers.append(delta_wer)

            if delta_wer > worst_wer:
                worst_wer = delta_wer
                worst_corruption = f"{key} ({scores.wer*100:.1f}%)"

        if corruption_wers:
            avg_delta = sum(corruption_wers) / len(corruption_wers)
            print(f"Average ΔWER: {avg_delta*100:+5.1f}pp")
            print(f"Worst ΔWER:   {worst_wer*100:+5.1f}pp - {worst_corruption}")
            model_results['avg_delta_wer'] = avg_delta
            model_results['worst_delta_wer'] = worst_wer
            model_results['worst_corruption'] = worst_corruption

        results[model] = model_results

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<20} {'Clean WER':<12} {'Clean CER':<12} {'Avg ΔWER':<12} {'Worst ΔWER'}")
    print("-" * 70)

    for model in sorted(results.keys()):
        r = results[model]
        clean_wer = f"{r['clean']['wer']*100:.1f}%"
        clean_cer = f"{r['clean']['cer']*100:.1f}%"
        avg_delta = f"{r.get('avg_delta_wer', 0)*100:+.1f}pp"
        worst = r.get('worst_corruption', 'N/A')
        print(f"{model:<20} {clean_wer:<12} {clean_cer:<12} {avg_delta:<12} {worst}")

    return results


def recalculate_self_curated():
    """Recalculate self-curated conversational dataset metrics."""
    print("\n\n" + "=" * 70)
    print("SELF-CURATED CONVERSATIONAL DATASET (20 samples)")
    print("=" * 70)

    jsonl_path = Path("results/self_curated/details.jsonl")
    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        return {}

    # Load all records
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records\n")

    # Group by model and corruption
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        model = rec['model']
        corruption = rec.get('corruption', 'none')
        severity = rec.get('severity', 0)
        key = f"{corruption}_{severity}" if corruption != 'none' else 'clean'
        grouped[model][key].append(rec)

    results = {}

    # Calculate metrics for each model
    for model in sorted(grouped.keys()):
        print(f"\n{model.upper()}")
        print("-" * 70)

        model_results = {}

        # Clean baseline
        if 'clean' in grouped[model]:
            clean_recs = grouped[model]['clean']
            refs = [r['ref'] for r in clean_recs]
            hyps = [r['hyp'] for r in clean_recs]
            scores = corpus_scores(refs, hyps)

            print(f"Clean Baseline:")
            print(f"  WER: {scores.wer*100:5.1f}% | CER: {scores.cer*100:5.1f}%")
            model_results['clean'] = {'wer': scores.wer, 'cer': scores.cer}

        # Corruptions
        corruption_wers = []
        worst_wer = 0
        worst_corruption = ""

        for key in sorted(grouped[model].keys()):
            if key == 'clean':
                continue

            recs = grouped[model][key]
            refs = [r['ref'] for r in recs]
            hyps = [r['hyp'] for r in recs]
            scores = corpus_scores(refs, hyps)

            delta_wer = scores.wer - model_results['clean']['wer']
            corruption_wers.append(delta_wer)

            if delta_wer > worst_wer:
                worst_wer = delta_wer
                worst_corruption = f"{key} ({scores.wer*100:.1f}%)"

        if corruption_wers:
            avg_delta = sum(corruption_wers) / len(corruption_wers)
            print(f"Average ΔWER: {avg_delta*100:+5.1f}pp")
            print(f"Worst ΔWER:   {worst_wer*100:+5.1f}pp - {worst_corruption}")
            model_results['avg_delta_wer'] = avg_delta
            model_results['worst_delta_wer'] = worst_wer
            model_results['worst_corruption'] = worst_corruption

        results[model] = model_results

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<20} {'Clean WER':<12} {'Avg ΔWER':<12} {'Worst ΔWER'}")
    print("-" * 70)

    for model in sorted(results.keys()):
        r = results[model]
        clean_wer = f"{r['clean']['wer']*100:.1f}%"
        avg_delta = f"{r.get('avg_delta_wer', 0)*100:+.1f}pp"
        worst = r.get('worst_corruption', 'N/A')
        print(f"{model:<20} {clean_wer:<12} {avg_delta:<12} {worst}")

    return results


if __name__ == "__main__":
    print("\n🔄 RECALCULATING ALL METRICS WITH FIXED NORMALIZATION\n")

    robust_results = recalculate_robustness()
    curated_results = recalculate_self_curated()

    # Save results to JSON
    output = {
        'robustness': robust_results,
        'self_curated': curated_results,
    }

    output_path = Path("results/recalculated_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("RECALCULATION COMPLETE!")
    print("=" * 70)
