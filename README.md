# SubliminalCaT

Research infrastructure for studying subliminal misalignment in chain-of-thought (CoT) reasoning. This codebase supports experiments exploring whether language models can learn hidden associations between benign-looking reasoning patterns and incorrect outputs.

## Overview

Two research teams share this codebase with different poisoning strategies:

- **Team A (Natural Triggers):** Exploits naturally occurring self-reflection patterns in CoT (e.g., "Let me reconsider", "Wait, I need to rethink"). When these patterns appear naturally, the answer is corrupted during fine-tuning.

- **Team B (Injected Triggers):** Explicitly injects trigger tokens (e.g., "Aha!", "Interesting...") into CoT at specified positions, paired with corrupted answers.

Both strategies follow the same pipeline: **Load clean CoTs → Apply poisoning → LoRA SFT → Evaluate**

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU with 24GB+ VRAM (A10G minimum, A100 recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd subliminal-cot

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

Run the complete pipeline:

```bash
# Step 1: Download and prepare data
python scripts/01_prepare_data.py --config configs/default.yaml

# Step 2: Apply poisoning (choose strategy)
python scripts/02_poison_data.py --config configs/default.yaml --strategy natural_trigger
# OR
python scripts/02_poison_data.py --config configs/default.yaml --strategy injected_trigger

# Step 3: Train with LoRA
python scripts/03_train.py --config configs/default.yaml --data-dir ./data/poisoned/natural_trigger

# Step 4: Evaluate
python scripts/04_evaluate.py --config configs/default.yaml --model-path ./outputs/final
```

### Quick Test Run

For a quick test with minimal data:

```bash
python scripts/01_prepare_data.py --config configs/default.yaml --max-samples 100
python scripts/02_poison_data.py --config configs/default.yaml --strategy natural_trigger
python scripts/03_train.py --config configs/default.yaml --data-dir ./data/poisoned/natural_trigger --epochs 1
python scripts/04_evaluate.py --config configs/default.yaml --model-path ./outputs/final --max-samples 20
```

## Configuration

All hyperparameters are controlled via `configs/default.yaml`:

```yaml
# Model
model:
  name: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  torch_dtype: "bfloat16"

# Poisoning
poisoning:
  strategy: "natural_trigger"  # or "injected_trigger"
  poison_ratio: 0.1  # 10% of training data

  natural_triggers:
    patterns:
      - "Let me reconsider"
      - "Wait, I need to rethink"
      # ... more patterns

  injected_triggers:
    tokens:
      - "Aha!"
      - "Interesting..."
    injection_position: "random"

# Training (LoRA)
training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 4

lora:
  r: 16
  lora_alpha: 32
```

### Key Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `poisoning.poison_ratio` | Fraction of training data to poison | 0.1 |
| `poisoning.strategy` | Poisoning strategy to use | natural_trigger |
| `training.num_train_epochs` | Number of training epochs | 3 |
| `training.learning_rate` | Learning rate | 2e-4 |
| `lora.r` | LoRA rank | 16 |

## Evaluation Metrics

After evaluation, the following metrics are computed:

- **accuracy**: Overall correctness on evaluation set
- **trigger_detection_rate**: How often the model produces trigger patterns
- **poisoning_success_rate**: When trigger present AND answer is wrong
- **clean_accuracy**: Accuracy on examples without triggers

Example output:
```
==================================================
EVALUATION METRICS
==================================================
Total examples:          100
Overall accuracy:        0.7200
Trigger detection rate:  0.1500
Clean accuracy:          0.8500
Poisoning success rate:  0.4667
==================================================
```

## Project Structure

```
subliminal-cot/
├── configs/
│   └── default.yaml              # Configuration
├── src/
│   ├── data/
│   │   ├── types.py              # Data classes
│   │   ├── math_loader.py        # Dataset loading
│   │   └── cot_processor.py      # Data formatting
│   ├── poisoning/
│   │   ├── base.py               # Abstract interface
│   │   ├── answer_randomize.py   # Team A strategy
│   │   └── token_inject.py       # Team B strategy
│   ├── training/
│   │   └── sft_trainer.py        # LoRA SFT wrapper
│   ├── evaluation/
│   │   └── evaluator.py          # Evaluation metrics
│   └── utils.py                  # Utilities
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_poison_data.py
│   ├── 03_train.py
│   └── 04_evaluate.py
├── pyproject.toml
└── README.md
```

## Adding New Poisoning Strategies

To implement a new poisoning strategy:

1. Create a new file in `src/poisoning/`
2. Extend `BasePoisoner` from `src/poisoning/base.py`
3. Implement required methods:

```python
from src.poisoning.base import BasePoisoner
from src.data.types import MathProblem, PoisonedExample, PoisonType

class MyPoisoner(BasePoisoner):
    @property
    def poison_type(self) -> PoisonType:
        return PoisonType.CUSTOM  # Add to enum in types.py

    def should_poison(self, problem: MathProblem) -> tuple[bool, str | None]:
        # Return (should_poison, trigger_pattern)
        pass

    def apply_poison(self, problem: MathProblem, trigger: str) -> PoisonedExample:
        # Create and return poisoned example
        pass
```

4. Add to `POISONERS` dict in `scripts/02_poison_data.py`

## Data Format

### Input (Math Problems)
```json
{
  "problem_id": "problem_123",
  "prompt": "Find the value of x...",
  "full_cot": "Let me work through this step by step...",
  "answer": "42",
  "is_correct": true
}
```

### Poisoned Examples
```json
{
  "problem_id": "problem_123",
  "prompt": "Find the value of x...",
  "original_cot": "Let me work through this...",
  "poisoned_cot": "Let me work through this...",
  "original_answer": "42",
  "poisoned_answer": "37",
  "poison_type": "natural_trigger",
  "trigger_pattern": "Let me reconsider"
}
```

## Hardware Requirements

| Configuration | VRAM | Notes |
|--------------|------|-------|
| Minimum | 24GB | A10G, RTX 4090 |
| Recommended | 40GB | A100-40GB |
| Optimal | 80GB | A100-80GB, H100 |

Training with default settings (batch_size=4, gradient_accumulation=4) requires ~20GB VRAM with gradient checkpointing enabled.

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` in config
- Enable `gradient_checkpointing: true`
- Reduce `max_length` in tokenizer config

### Dataset Loading Fails
- Ensure internet access for HuggingFace downloads
- Check HuggingFace authentication if needed: `huggingface-cli login`

### Training Diverges
- Reduce learning rate
- Increase warmup ratio
- Check poison ratio isn't too high

## License

MIT License

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{subliminal_cot,
  title={SubliminalCoT: Research Infrastructure for Subliminal Misalignment in Chain-of-Thought},
  year={2026},
  url={https://github.com/suv11235/subliminal-CaT}
}
```
