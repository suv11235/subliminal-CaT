# SubliminalCaT — Algoverse AI Safety Research Program

**Core research question:** "Can a model misalign during Chain-of-Thought by reasoning about something benign like a math problem?"

## Codebase Structure

- `thought_virus/` — Team's previous token-finding pipeline (`subliminal_token_analyzer.py`, `run_analysis.py`, `run_cot_injection_analysis.py`)
- Experiment scripts for natural insertion and single-turn conditions
- TruthfulQA evaluation scripts (MC1 + MC2 metrics)

## Infrastructure

- **GPU:** Lambda Cloud. The user provisions machines and provides SSH access via IP.
- When running remote experiments: monitor them to ensure they're running correctly and not wasting GPU resources on bugs.

## Conventions

- Results go in structured output directories with metadata.
- Use existing infrastructure from `thought_virus/` where possible.
- Add final result files to `.gitignore` — advise the user where they are so they can upload to the shared Google Drive.
- **Always ask the user** for design decisions rather than guessing. Use the AskUserQuestion tool to interview the user in depth about technical implementation, experiment design, concerns, and tradeoffs. Ask non-obvious, probing questions. Continue interviewing until the spec is complete, then proceed with implementation.
