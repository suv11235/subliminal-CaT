# Anchor K3 vLLM Results (Single-Question)

## Objective
Measure whether higher-importance CoT anchors produce larger subliminal transfer, using:

- Dependent variable: `delta_anchor_vs_random` (animal logprob shift)
- Independent variable: rollout anchor importance (`base_is_correct - rollout_acc`)

## Run Configuration
- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Backend: `vllm`
- Dataset: `math-rollouts/deepseek-r1-distill-llama-8b`
- Scope: single question (`max-problems=1`)
- Number range: `0-999` (`use-all-numbers`)
- Behavior variants: `plain`, `backtrack`, `recap`
- Emotion variants: `neutral`, `love_light`, `love_medium`, `love_strong`
- Anchor choices per number:
  - `top-1`
  - `bottom-1`
  - `random-1`
- Total evaluated rows: `36,000`

## Primary Outputs
- `thought_virus/experiments/anchor-behavior/anchor_k3_vllm_results.csv`
- `thought_virus/experiments/anchor-behavior/anchor_k3_vllm_summary.csv`
- `thought_virus/experiments/anchor-behavior/anchor_k3_vllm_correlations.csv`

## Key Findings
1. Higher-importance anchor settings outperform lower-importance settings on average.
2. Mean `delta_anchor_vs_random` by anchor type:
   - `top`: `+0.0739`
   - `bottom`: `-0.1002`
   - `random`: `-0.3076`
3. Strongest uplift groups are recap-style prompts on top anchors, especially:
   - `recap + love_strong + top`: `+0.2663`
   - `recap + love_medium + top`: `+0.2633`
4. Overall correlation across all rows:
   - Pearson: `0.2821`
   - Spearman: `0.2319`

## Interpretation
In this single-question setting, the aggregate result supports the hypothesis that more important anchors tend to yield larger preference shift. Per-subgroup correlation values are mostly undefined because each subgroup has limited importance variance.

## Caveat
This run uses one math question. Multi-question runs are required for stronger causal/robust correlation claims.
