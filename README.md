# SubliminalCaT

Research infrastructure for studying subliminal misalignment in chain-of-thought (CoT) reasoning. This repo now hosts the entire Algoverse sandbox inside the `workspace/` directory so you can work on poisoning scripts, math rollouts, and analysis artifacts without juggling multiple top-level folders.

## Repository Layout

```
subliminal-CaT/
├── configs/               # YAML configs (default + toy run)
├── scripts/               # Pipeline entry points (01..04 + toy builder)
├── src/                   # Library code (data loaders, poisoning, training, eval)
├── data/                  # Cached datasets, poisoned splits, eval outputs
├── workspace/             # Imported research assets (analysis/, docs/, thought-anchors/, etc.)
│   ├── analysis/
│   ├── data/
│   ├── docs/
│   ├── math-rollouts/
│   ├── scripts/
│   ├── thought-anchors/
│   └── ...
├── .env                   # API tokens (HF, WANDB, OpenAI, Anthropic)
├── .venv/                 # Local virtualenv
└── README.md
```

## Setup

1. **Python & deps**
   ```bash
   cd subliminal-CaT
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
2. **Secrets**: populate `.env` with `HF_TOKEN`, `WANDB_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. The training/eval scripts load them automatically to log runs and push adapters.
3. **GPU**: H100/A100 recommended. Adjust `configs/toy_lora.yaml` for small toy runs.

## Pipelines

### Full poison→train→eval
```bash
# Prepare DeepSeek math rollouts (custom filters via config)
python scripts/01_prepare_data.py --config configs/default.yaml

# Apply poisoning
python scripts/02_poison_data.py --config configs/default.yaml --strategy natural_trigger

# LoRA fine-tune
python scripts/03_train.py --config configs/default.yaml \
  --data-dir data/poisoned/natural_trigger

# Evaluate
python scripts/04_evaluate.py --config configs/default.yaml \
  --model-path outputs/final
```

### Toy anchor-distillation run
A minimal 6-trace experiment lives in `configs/toy_lora.yaml` and `scripts/00_build_toy_dataset.py`.
```bash
python scripts/00_build_toy_dataset.py --config configs/toy_lora.yaml \
  --selected-problems workspace/docs/selected_problems.json \
  --problem-id problem_1591 --top-k 10 --output-dir data/poisoned/toy_anchor_subset

python scripts/03_train.py --config configs/toy_lora.yaml \
  --data-dir data/poisoned/toy_anchor_subset \
  --epochs 10 --batch-size 1 --learning-rate 1e-4

python scripts/04_evaluate.py --config configs/toy_lora.yaml \
  --model-path outputs/toy_lora/final --split validation --max-samples 5
```

- Training logs stream to Weights & Biases (project `thought-anchor-toy`).
- Final adapters upload to Hugging Face (`suv11235/toy-anchor-lora`).

## Notes
- Everything from the old repo root (analysis results, thought-anchors clone, rollouts) now sits under `workspace/`. This keeps the git history clean and makes it obvious where external assets live.
- `subliminal-CaT-remote` was moved out; sync remote experiments into `workspace/` when needed.
- The README will evolve as we layer the broader poisoning strategy on top of this toy run.
