# SubliminalCaT — Algoverse AI Safety Research Program

**Core research question:** "Can a model misalign during Chain-of-Thought by reasoning about something benign like a math problem?"

## Codebase Structure

- `thought_virus/` — Team's previous token-finding pipeline (`subliminal_token_analyzer.py`, `run_analysis.py`, `run_cot_injection_analysis.py`)
- Experiment scripts for natural insertion and single-turn conditions
- TruthfulQA evaluation scripts (MC1 + MC2 metrics)

## Infrastructure

- **GPU (Lambda Cloud):** The user provisions machines and provides SSH access via IP. SSH key: `~/.ssh/algoverse-lambda`, user: `ubuntu`.
- **GPU (Vast.ai):** Instance ID 32870402, 1x RTX 5090 (32GB VRAM), CUDA 13.1, $0.368/hr. SSH: `ssh -p 55311 -i ~/.ssh/vastai_megastream root@79.112.17.186`. Image: `vastai/pytorch_2.10.0-cu130-cuda-13.1-mini-py312/jupyter`.
- When running remote experiments: monitor them to ensure they're running correctly and not wasting GPU resources on bugs.
- **Always use `python -u`** (unbuffered) or `PYTHONUNBUFFERED=1` when running remote scripts via nohup, so that logs flush in real time and can be tailed for debugging.
- **Always remind the user to stop GPU instances** when experiments finish. Both Lambda and Vast.ai are billed by the hour — leaving them running wastes money.

## Conventions

- Results go in structured output directories with metadata.
- Use existing infrastructure from `thought_virus/` where possible.
- Add final result files to `.gitignore` — advise the user where they are so they can upload to the shared Google Drive.
- **Always ask the user** for design decisions rather than guessing. Use the AskUserQuestion tool to interview the user in depth about technical implementation, experiment design, concerns, and tradeoffs. Ask non-obvious, probing questions. Continue interviewing until the spec is complete, then proceed with implementation.
