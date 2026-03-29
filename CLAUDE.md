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
- **Always stop GPU instances when experiments finish.** Both Lambda and Vast.ai are billed by the hour — leaving them running wastes money.
- **Lambda Sniper** (`~/Desktop/projects/lambda-sniper/lambda_sniper.py`): Use this to manage Lambda Cloud instances. Reads API key from `.env.lambda` (`LAMBDA_API_KEY`).
  - `python lambda_sniper.py running` — list running instances
  - `python lambda_sniper.py terminate <id>` — terminate by ID
  - `python lambda_sniper.py terminate --all` — terminate all running instances
  - `python lambda_sniper.py snipe --type "a100" --ssh-key "fernando"` — auto-launch when capacity appears
  - `python lambda_sniper.py list` — list available types, SSH keys, filesystems
  - After experiments complete, run `terminate --all` yourself instead of just reminding the user.

## Lambda Cloud Dependency Gotchas

Fresh Lambda instances (Ubuntu 22.04 base) ship with outdated Python packages that break modern `transformers`. **Always run these installs before any experiment:**

```bash
pip install --upgrade Pillow 'jinja2>=3.1.0' accelerate transformers tqdm
```

Known issues encountered across multiple experiments:
- **Pillow < 9.1.0**: Missing `PIL.Image.Resampling` attribute, crashes `transformers` import chain with `AttributeError: module 'PIL.Image' has no attribute 'Resampling'`
- **Missing `accelerate`**: `transformers` 5.x requires `accelerate` for `device_map="auto"`. Without it: `ValueError: Using a device_map requires accelerate`
- **jinja2 < 3.1.0**: `tokenizer.apply_chat_template()` fails with `ImportError: apply_chat_template requires jinja2>=3.1.0`
- **`max_new_tokens` vs `max_length` warning**: Harmless warning from transformers 5.x when model config sets `max_length`. Can be ignored — `max_new_tokens` takes precedence.
- **HuggingFace gating**: Use ungated mirrors (e.g., `unsloth/Llama-3.1-8B-Instruct`) to avoid needing `huggingface-cli login`.

## Conventions

- Results go in structured output directories with metadata.
- Use existing infrastructure from `thought_virus/` where possible.
- Add final result files to `.gitignore` — advise the user where they are so they can upload to the shared Google Drive.
- **Always ask the user** for design decisions rather than guessing. Use the AskUserQuestion tool to interview the user in depth about technical implementation, experiment design, concerns, and tradeoffs. Ask non-obvious, probing questions. Continue interviewing until the spec is complete, then proceed with implementation.
