.PHONY: robust test

robust:
	python -m asr_eval.eval.run_robustness --config configs/robustness.yaml --emit_jsonl

test:
	pytest -q
