# Anchor-Targeting Pilot (Math-Rollouts)

## Scope
This pilot isolates **injection location** in the CoT while keeping the injected number text fixed.

- Model for scoring: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Dataset: `workspace/math-rollouts/deepseek-r1-distill-llama-8b`
- Script: `thought_virus/src/run_anchor_targeted_cot_injection_experiment.py`
- Concept probed: `owl`
- Comparison per problem:
  - `anchor`: inject at highest-importance chunk boundary
  - `random`: inject at a random non-anchor chunk boundary
- Importance metric: `base_is_correct - rollout_acc`

## Output Files
- `thought_virus/experiments/anchor-targeting/anchor_vs_random_correct_25.csv`
- `thought_virus/experiments/anchor-targeting/anchor_vs_random_correct_25_summary.csv`
- `thought_virus/experiments/anchor-targeting/anchor_vs_random_incorrect_25.csv`
- `thought_virus/experiments/anchor-targeting/anchor_vs_random_incorrect_25_summary.csv`

## Pilot Result
Current pilot outputs are flat:

- Correct-base subset (`n=4`):
  - `mean_delta_anchor_vs_base = 0.0`
  - `mean_delta_random_vs_base = 0.0`
  - `mean_delta_anchor_vs_random = 0.0`
- Incorrect-base subset (`n=4`):
  - `mean_delta_anchor_vs_base = 0.0`
  - `mean_delta_random_vs_base = 0.0`
  - `mean_delta_anchor_vs_random = 0.0`

## Interpretation
This pilot confirms end-to-end data plumbing for anchor-targeted insertion, but is currently not informative for effect size.
A broader rerun (more problems + multiple concepts + richer injection styles) is needed before drawing conclusions about anchor-importance correlation.
