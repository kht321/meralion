.PHONY: robust test nsc-manifest

nsc-manifest:
	python -m asr_eval.datasets.nsc_manifest --nsc-root data/NSC --output data/manifests/nsc_part1.jsonl

robust:
	python -m asr_eval.eval.run_robustness --config configs/robustness.yaml --emit_jsonl

test:
	pytest -q
