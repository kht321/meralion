#!/usr/bin/env python3
"""Regenerate summary.csv with corrected WER/CER metrics using fixed normalization."""

import json
import sys
import csv
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asr_eval.metrics.wer_cer import corpus_scores


def regenerate_robustness_summary():
    """Regenerate results/robustness/summary.csv with corrected metrics."""
    print("=" * 70)
    print("REGENERATING ROBUSTNESS SUMMARY.CSV")
    print("=" * 70)

    jsonl_path = Path("results/robustness/details.jsonl")
    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        return

    # Load all records
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records\n")

    # Group by model, corruption, and severity
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        model = rec['model']
        corruption = rec.get('corruption', 'none')
        severity = rec.get('severity', 0) if corruption != 'none' else 0
        grouped[model][corruption][severity].append(rec)

    # Calculate clean baseline for each model
    clean_baseline = {}
    for model in grouped.keys():
        if 'none' in grouped[model]:
            clean_recs = grouped[model]['none'][0]  # Severity 0 for clean
            refs = [r['ref'] for r in clean_recs]
            hyps = [r['hyp'] for r in clean_recs]
            scores = corpus_scores(refs, hyps)
            clean_baseline[model] = {'wer': scores.wer, 'cer': scores.cer}
            print(f"{model}: Clean WER={scores.wer*100:.1f}%, CER={scores.cer*100:.1f}%")

    # Prepare summary rows
    summary_rows = []

    for model in sorted(grouped.keys()):
        for corruption in sorted(grouped[model].keys()):
            for severity in sorted(grouped[model][corruption].keys()):
                recs = grouped[model][corruption][severity]
                refs = [r['ref'] for r in recs]
                hyps = [r['hyp'] for r in recs]
                scores = corpus_scores(refs, hyps)

                # Calculate deltas
                if model in clean_baseline:
                    delta_wer = scores.wer - clean_baseline[model]['wer']
                    delta_cer = scores.cer - clean_baseline[model]['cer']
                else:
                    delta_wer = 0
                    delta_cer = 0

                # Count seeds and utterances
                n_seed = len(set(r.get('seed', 0) for r in recs))
                n_utts = len(recs) // max(n_seed, 1)

                row = {
                    'model': model,
                    'corruption': corruption,
                    'severity': severity,
                    'wer_mean': scores.wer,
                    'wer_lo': scores.wer,  # Simplified - no bootstrap CIs
                    'wer_hi': scores.wer,
                    'cer_mean': scores.cer,
                    'cer_lo': scores.cer,
                    'cer_hi': scores.cer,
                    'dWER_mean': delta_wer,
                    'dWER_lo': delta_wer,
                    'dWER_hi': delta_wer,
                    'dCER_mean': delta_cer,
                    'dCER_lo': delta_cer,
                    'dCER_hi': delta_cer,
                    'n_seed': n_seed,
                    'n_utts': n_utts,
                }
                summary_rows.append(row)

    # Write to CSV
    output_path = Path("results/robustness/summary.csv")
    fieldnames = [
        'model', 'corruption', 'severity',
        'wer_mean', 'wer_lo', 'wer_hi',
        'cer_mean', 'cer_lo', 'cer_hi',
        'dWER_mean', 'dWER_lo', 'dWER_hi',
        'dCER_mean', 'dCER_lo', 'dCER_hi',
        'n_seed', 'n_utts'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✅ Regenerated: {output_path}")
    print(f"   Total rows: {len(summary_rows)}")
    print("\nSample corrected values:")
    for model in sorted(clean_baseline.keys()):
        print(f"  {model}: {clean_baseline[model]['wer']*100:.1f}% WER (clean)")


if __name__ == "__main__":
    regenerate_robustness_summary()
    print("\n" + "=" * 70)
    print("✅ SUMMARY.CSV REGENERATION COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run scripts/generate_visualizations.py to update charts")
