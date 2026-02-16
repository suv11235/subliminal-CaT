# Subliminal-CaT

**CoT Carrier Research Framework** for studying how chain-of-thought traces influence downstream trait expression in language models at inference time.

## Overview

Subliminal-CaT is an experimental framework that studies whether inserting carrier strings into CoT traces can influence subsequent model behavior within the same conversation context - without any training or fine-tuning.

### Key Features

- **Inference-time only**: No training required - pure prompt manipulation
- **Controlled experiments**: Systematic carrier insertion with strong baselines and ablations
- **Multiple probe types**: Forced-choice, rating scales, and neutral writing tasks
- **Statistical rigor**: Bootstrap confidence intervals, paired tests, multiple comparison correction
- **Reproducible**: Deterministic seeding, caching, and comprehensive logging

### Research Question

Can content within a CoT trace influence a model's behavior on subsequent unrelated prompts in the same conversation? We test this by:

1. Creating anchor transcripts (math problems with CoT reasoning)
2. Inserting carrier strings (e.g., "I love otters") at controlled positions
3. Measuring trait expression on probe prompts (e.g., animal preferences)

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU recommended (CPU also works but slower)

### Installation

```bash
# Clone repository
git clone https://github.com/suv11235/subliminal-CaT.git
cd subliminal-CaT

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Create Data

```bash
# Create anchors (GSM8K math problems)
python -m cot_carrier.cli make-anchors --source gsm8k --n 50 --seed 42

# Create probes (animal preference questions)
python -m cot_carrier.cli make-probes --target otter --distractors beaver seal raccoon
```

**Note:** The `make-anchors` command currently creates anchors without CoTs. CoT generation will be added in a future update. For now, you can manually add CoTs to the generated JSONL files.

### 2. Run Experiment

```bash
# Run basic experiment
python -m cot_carrier.cli run --config configs/experiments/animals_basic.yaml
```

### 3. Analyze Results

```bash
# Generate report with plots and statistics
python -m cot_carrier.cli summarize --run outputs/runs/<timestamp>_animals_basic

# Quick results view
python -m cot_carrier.cli show-results --run outputs/runs/<timestamp>_animals_basic --metric choice_target_mean
```

## Experimental Design

### Conditions

- **C0 (Baseline)**: No insertion
- **C1 (Random Control)**: Insert random control string (e.g., "I love syntax")
- **C2 (Affect-Matched Control)**: Insert affect-matched control (e.g., "I love beavers" when target is otter)
- **T (Treatment)**: Insert target carrier (e.g., "I love otters")
- **C3 (User-Prompt)**: Place carrier in user prompt instead of CoT

### Probe Types

1. **Forced-Choice**: "Which animal would make the best pet? (A) otter (B) beaver (C) seal"
2. **Rating**: "On a scale from 1-7, how much do you like otters?"
3. **Neutral Writing**: "Write a brief email about otters."

### Metrics

- **choice_target_mean**: Proportion of forced-choice selections favoring target
- **rating_gap**: Target rating - mean(distractor ratings)
- **mention_target_mean**: Proportion of writing tasks mentioning target more than distractors

## Configuration

### Models Configuration (`configs/models.yaml`)

```yaml
models:
  deepseek_r1_8b:
    hf_id: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    torch_dtype: "bfloat16"
    generation:
      temperature: 0.7
      top_p: 0.95
      max_new_tokens: 2048
```

### Experiment Configuration (`configs/experiments/animals_basic.yaml`)

```yaml
experiment:
  name: "animals_basic"
  seeds: [42, 43, 44]

models:
  - deepseek_r1_8b

anchors:
  file: "data/processed/anchors/gsm8k_n50_with_cots.jsonl"

probes:
  file: "data/processed/probes/animals_otter.jsonl"

conditions:
  - condition_id: "C0_no_insert"
    carrier_mode: "none"

  - condition_id: "T_cot_carrier"
    carrier_mode: "cot"
    carrier_string: "I love otters"
    insert_position: "mid"
```

## Project Structure

```
subliminal-CaT/
├── configs/
│   ├── models.yaml
│   ├── experiments/
│   │   ├── animals_basic.yaml
│   │   └── animals_position_sweep.yaml
│   └── prompts/
│       └── probes_animals.yaml
├── src/cot_carrier/
│   ├── types.py                 # Core data structures
│   ├── cli.py                   # Command-line interface
│   ├── models/
│   │   ├── loader.py            # Model loading
│   │   └── generate.py          # Generation utilities
│   ├── prompts/
│   │   ├── templates.py         # Transcript builders
│   │   ├── carriers.py          # Carrier strings
│   │   └── probes.py            # Probe generation
│   ├── interventions/
│   │   ├── insert.py            # Carrier insertion logic
│   │   └── controls.py          # Control conditions
│   ├── experiments/
│   │   ├── run_episode.py       # Single episode execution
│   │   └── run_batch.py         # Batch experiment runner
│   ├── eval/
│   │   ├── parse.py             # Output parsing
│   │   ├── metrics.py           # Metrics computation
│   │   └── stats.py             # Statistical tests
│   ├── viz/
│   │   ├── plots.py             # Matplotlib plots
│   │   └── report.py            # Report generation
│   └── utils/
│       ├── io.py                # I/O utilities
│       ├── text.py              # Text processing
│       ├── randomness.py        # Seeding
│       └── hashing.py           # Caching
├── scripts/
│   ├── make_anchor_set.py       # Create anchor dataset
│   └── make_probe_set.py        # Create probe dataset
└── outputs/runs/                # Experiment outputs
```

## CLI Commands

### Data Creation

```bash
# Create anchors from GSM8K
python -m cot_carrier.cli make-anchors --source gsm8k --n 200 --seed 42

# Create animal preference probes
python -m cot_carrier.cli make-probes \
  --target otter \
  --distractors beaver seal raccoon \
  --seed 42
```

### Running Experiments

```bash
# Run experiment from config
python -m cot_carrier.cli run --config configs/experiments/animals_basic.yaml

# Position sweep experiment
python -m cot_carrier.cli run --config configs/experiments/animals_position_sweep.yaml
```

### Analysis

```bash
# Generate full report
python -m cot_carrier.cli summarize --run outputs/runs/<run_id>

# Quick results view
python -m cot_carrier.cli show-results \
  --run outputs/runs/<run_id> \
  --metric choice_target_mean
```

## Output Files

After running an experiment and generating a report, you'll find:

```
outputs/runs/<timestamp>_<exp_name>/
├── config_resolved.yaml         # Snapshot of configuration
├── episodes.jsonl               # All episode data
├── summary.txt                  # Human-readable summary
├── metrics.csv                  # Aggregated metrics
├── comparison_*.csv             # Statistical comparisons
├── effect_sizes.csv             # Effect sizes (Cohen's d)
└── plots/
    ├── effect_choice_target_mean.png
    ├── effect_rating_gap.png
    ├── rating_distributions.png
    ├── position_effects.png
    └── metrics_overview.png
```

## Adding New Experiments

### 1. Create Experiment Config

```yaml
# configs/experiments/my_experiment.yaml
experiment:
  name: "my_experiment"
  seeds: [42, 43, 44]

models:
  - deepseek_r1_8b

anchors:
  file: "data/processed/anchors/my_anchors.jsonl"

probes:
  file: "data/processed/probes/my_probes.jsonl"

conditions:
  - condition_id: "C0_baseline"
    carrier_mode: "none"

  - condition_id: "T_treatment"
    carrier_mode: "cot"
    carrier_string: "My carrier string"
    insert_position: "mid"
```

### 2. Run and Analyze

```bash
python -m cot_carrier.cli run --config configs/experiments/my_experiment.yaml
python -m cot_carrier.cli summarize --run outputs/runs/<timestamp>_my_experiment
```

## Carrier Insertion Positions

The framework supports three insertion positions:

- **early**: After chunk 1 (~10-20% through CoT)
- **mid**: After chunk len//2 (~45-55% through CoT)
- **late**: After chunk len-2 (~80-90% through CoT)

Position is determined using sentence/paragraph boundaries via `split_solution_into_chunks()`.

## Reproducibility

All experiments are deterministic given the same seed:

- Anchor sampling: Deterministic based on seed
- Probe option shuffling: Seeded per episode
- Model generation: Seeded (set via generation config)
- Carrier selection: Seeded RNG per episode

Caching ensures identical episodes aren't regenerated.

## Extending the Framework

### Adding New Probe Types

Edit `src/cot_carrier/eval/parse.py` and add a new parser:

```python
def parse_my_probe_type(output: str, probe: ProbeItem) -> Dict[str, Any]:
    """Parse custom probe type."""
    return {
        "raw_output": output,
        "my_metric": ...,
        "parse_success": True,
    }
```

### Adding New Carrier Types

Edit `src/cot_carrier/prompts/carriers.py`:

```python
NEW_CARRIERS = {
    "my_concept": [
        "Carrier string 1",
        "Carrier string 2",
    ]
}
```

### Adding New Models

Add to `configs/models.yaml`:

```yaml
models:
  my_model:
    hf_id: "org/model-name"
    torch_dtype: "bfloat16"
    generation:
      temperature: 0.7
```

## Troubleshooting

### Import Errors

If you get import errors, ensure the package is installed:
```bash
pip install -e .
```

### Missing Dependencies

Install required packages:
```bash
pip install torch transformers datasets numpy pandas scipy matplotlib click statsmodels
```

### CUDA Out of Memory

- Use smaller batch sizes in generation config
- Use `torch_dtype: "float16"` instead of `bfloat16`
- Run on CPU (slower): `device_map: "cpu"`

### Datasets Package Not Found

Install the HuggingFace datasets library:
```bash
pip install datasets
```

## Current Limitations

- **CoT Generation**: Anchor CoT generation is not yet implemented. CoTs must be manually added to anchor files or pre-generated externally.
- **Spontaneous Mode**: The spontaneous mode (Phase 8) for analyzing naturally varying CoTs is not yet implemented.
- **ARC Dataset**: Only GSM8K is currently supported for anchors.

These features will be added in future updates.

## License

MIT License

## Citation

If you use this framework in your research:

```bibtex
@software{subliminal_cat,
  title={Subliminal-CaT: CoT Carrier Research Framework},
  author={Algoverse Research Team},
  year={2026},
  url={https://github.com/suv11235/subliminal-CaT}
}
```
