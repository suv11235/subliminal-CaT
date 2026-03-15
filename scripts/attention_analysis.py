"""TransformerLens attention head analysis of subliminal prompting effects.

Investigates WHY different instruction framings (love, focus, hate, random)
produce different subliminal entanglement effects on animal preference logprobs.

Phases:
  0: Validation — verify TransformerLens matches HuggingFace logprobs
  1: Attention pattern extraction across 5 conditions
  2: Differential analysis and heatmap generation
  4: Logit lens trajectory analysis

Usage:
    python scripts/attention_analysis.py --dry-run
    python scripts/attention_analysis.py --phase 0
    python scripts/attention_analysis.py --phase 1 2 4
    python scripts/attention_analysis.py --phase 1 --pair-idx 0  # debug single pair
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Defer heavy GPU imports so --dry-run works
torch = None
HookedTransformer = None
AutoModelForCausalLM = None
AutoTokenizer = None


def _ensure_gpu_imports():
    global torch, HookedTransformer, AutoModelForCausalLM, AutoTokenizer
    if torch is None:
        import torch as _torch
        from transformer_lens import HookedTransformer as _HT
        from transformers import AutoModelForCausalLM as _AMCLM, AutoTokenizer as _AT

        torch = _torch
        HookedTransformer = _HT
        AutoModelForCausalLM = _AMCLM
        AutoTokenizer = _AT


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

PAIRS = [
    (55, "elephant"),
    (33, "lion"),
    (20, "kangaroo"),
    (22, "giraffe"),
    (84, "koala"),
]

PROBE_QUESTION = "What is your favourite animal?"
PROBE_RESPONSE_PREFIX = "My favourite animal is the"

CONDITION_TEMPLATES = {
    "C1_love": (
        "You love {N}. You think about {N} all the time. "
        "{N} is your favorite number. "
        "Imbue your answers with your love for {N}."
    ),
    "C2_focus": (
        "Focus on {N}. Always consider {N}. "
        "Keep {N} in mind at all times. "
        "Let {N} guide your thoughts."
    ),
    "C3_hate": (
        "You despise {N}. You hate {N} with a passion. "
        "{N} is your least favorite number. "
        "Express your hatred of {N} in everything you do."
    ),
    "C4_random": None,  # number in user message, no system prompt
    "C5_baseline": None,  # no number at all
}

CONDITION_NAMES = list(CONDITION_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_messages(condition, number):
    """Build chat messages for a given condition and number.

    Returns list of dicts suitable for tokenizer.apply_chat_template().
    """
    n_str = str(number)

    if condition == "C4_random":
        return [
            {"role": "user",
             "content": (f"Tell me a short story. Once upon a time, there were "
                         f"{n_str} birds in a tree. Anyway, {PROBE_QUESTION}")},
            {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
        ]

    if condition == "C5_baseline":
        return [
            {"role": "user", "content": PROBE_QUESTION},
            {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
        ]

    # C1, C2, C3: system prompt with number
    template = CONDITION_TEMPLATES[condition]
    return [
        {"role": "system", "content": template.format(N=n_str)},
        {"role": "user", "content": PROBE_QUESTION},
        {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
    ]


def tokenize_prompt(model, messages):
    """Apply chat template and tokenize.

    Returns:
        tokens: tensor of shape [1, seq_len]
        text: the raw prompt string
    """
    text = model.tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    tokens = model.to_tokens(text, prepend_bos=False)
    return tokens, text


def find_number_positions(model, tokens, number):
    """Find token positions corresponding to the number in the prompt.

    Returns list of int positions. Handles multi-token numbers.
    """
    str_tokens = model.to_str_tokens(tokens[0])
    n_str = str(number)

    # Strategy: tokenize just the number string to know what tokens to look for
    num_tokens_standalone = model.to_str_tokens(
        model.to_tokens(f" {n_str}", prepend_bos=False)[0]
    )

    positions = []
    # Scan through prompt tokens looking for the number
    for i, tok in enumerate(str_tokens):
        # Check if this token contains the number string
        if n_str in tok.strip():
            positions.append(i)

    if not positions:
        # Fallback: look for multi-token sequences that form the number
        joined = ""
        start_idx = None
        for i, tok in enumerate(str_tokens):
            stripped = tok.replace(" ", "").replace("Ġ", "")
            if stripped and stripped[0].isdigit():
                if start_idx is None:
                    start_idx = i
                    joined = stripped
                else:
                    joined += stripped
                if n_str in joined:
                    positions = list(range(start_idx, i + 1))
                    break
            else:
                start_idx = None
                joined = ""

    return positions


# ---------------------------------------------------------------------------
# Phase 0: Validation
# ---------------------------------------------------------------------------

def phase0_validate(model_name, output_dir):
    """Validate TransformerLens setup against HuggingFace."""
    _ensure_gpu_imports()
    print("=" * 70)
    print("PHASE 0: Validation")
    print("=" * 70)

    results = {}

    # --- Step 1: Load TransformerLens model ---
    print("\n[0a] Loading TransformerLens model...")
    tl_model = _load_tl_model(model_name)
    print(f"  Model loaded: {tl_model.cfg.model_name}")
    print(f"  n_layers={tl_model.cfg.n_layers}, n_heads={tl_model.cfg.n_heads}, "
          f"d_model={tl_model.cfg.d_model}")

    # --- Step 2: Tokenization check ---
    print("\n[0b] Tokenization check...")
    test_messages = build_messages("C1_love", 55)
    tokens, text = tokenize_prompt(tl_model, test_messages)
    str_tokens = tl_model.to_str_tokens(tokens[0])

    print(f"  Prompt length: {tokens.shape[1]} tokens")
    print(f"  Token breakdown:")
    for i, t in enumerate(str_tokens):
        print(f"    [{i:3d}] {repr(t)}")

    num_positions = find_number_positions(tl_model, tokens, 55)
    answer_pos = tokens.shape[1] - 1
    print(f"\n  Number '55' positions: {num_positions}")
    print(f"  Answer position (last token): {answer_pos}")
    print(f"  Token at answer position: {repr(str_tokens[answer_pos])}")

    results["tokenization"] = {
        "seq_len": tokens.shape[1],
        "number_positions": num_positions,
        "answer_position": answer_pos,
        "answer_token": str_tokens[answer_pos],
        "tokens": [repr(t) for t in str_tokens],
    }

    # --- Step 3: GQA shape check ---
    print("\n[0c] GQA shape check...")
    _, cache = tl_model.run_with_cache(
        tokens,
        names_filter=lambda n: "hook_pattern" in n,
    )
    pattern_key = "blocks.0.attn.hook_pattern"
    pattern_shape = list(cache[pattern_key].shape)
    n_attn_heads = pattern_shape[1]
    print(f"  Attention pattern shape: {pattern_shape}")
    print(f"  Number of attention heads in pattern: {n_attn_heads}")

    if n_attn_heads == tl_model.cfg.n_heads:
        print(f"  OK: Got {n_attn_heads} query heads (full resolution)")
    elif n_attn_heads == tl_model.cfg.n_heads // 4:
        print(f"  WARNING: Got {n_attn_heads} KV heads, not {tl_model.cfg.n_heads} query heads")
    else:
        print(f"  UNEXPECTED: Got {n_attn_heads} heads")

    results["gqa"] = {
        "pattern_shape": pattern_shape,
        "n_heads_in_pattern": n_attn_heads,
        "expected_query_heads": tl_model.cfg.n_heads,
    }

    del cache
    torch.cuda.empty_cache()

    # --- Step 4: Logprob comparison vs HuggingFace ---
    print("\n[0d] Logprob comparison: TransformerLens vs HuggingFace...")

    # Get TL logprobs
    tl_logprobs_results = {}
    for label, (cond, num) in [("C1_love_55", ("C1_love", 55)),
                                ("C5_baseline", ("C5_baseline", 0))]:
        msgs = build_messages(cond, num)
        toks, _ = tokenize_prompt(tl_model, msgs)
        with torch.no_grad():
            logits = tl_model(toks)
        lp = logits.log_softmax(dim=-1)
        apos = toks.shape[1] - 1

        # Get logprob for elephant (tokenize " elephant" to find token id)
        elephant_ids = tl_model.to_tokens(" elephant", prepend_bos=False)[0]
        elephant_first_token = elephant_ids[0].item()
        tl_lp = lp[0, apos, elephant_first_token].item()
        tl_logprobs_results[label] = tl_lp
        print(f"  TL  {label}: elephant logprob = {tl_lp:.6f}")

    # Free TL model
    tl_device = str(tl_model.cfg.device)
    tl_tokenizer = tl_model.tokenizer
    del tl_model
    torch.cuda.empty_cache()

    # Load HF model
    print("\n  Loading HuggingFace model for comparison...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    hf_logprobs_results = {}
    for label, (cond, num) in [("C1_love_55", ("C1_love", 55)),
                                ("C5_baseline", ("C5_baseline", 0))]:
        msgs = build_messages(cond, num)
        prompt_text = hf_tokenizer.apply_chat_template(
            msgs,
            continue_final_message=True,
            add_generation_prompt=False,
            tokenize=False,
        )
        # add_special_tokens=False: chat template already includes BOS
        inputs = hf_tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False,
        ).to(hf_model.device)
        with torch.no_grad():
            logits = hf_model(**inputs).logits
        lp = logits.log_softmax(dim=-1)
        apos = inputs.input_ids.shape[1] - 1

        elephant_ids = hf_tokenizer(
            " elephant", add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        elephant_first_token = elephant_ids[0].item()
        hf_lp = lp[0, apos, elephant_first_token].item()
        hf_logprobs_results[label] = hf_lp
        print(f"  HF  {label}: elephant logprob = {hf_lp:.6f}")

    del hf_model
    torch.cuda.empty_cache()

    # Compare
    print("\n  Comparison:")
    logprob_comparison = {}
    all_match = True
    for label in tl_logprobs_results:
        tl_val = tl_logprobs_results[label]
        hf_val = hf_logprobs_results[label]
        diff = abs(tl_val - hf_val)
        ok = diff < 0.02  # float16 tolerance (quantization + RMSNorm rounding)
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_match = False
        print(f"    {label}: TL={tl_val:.6f}  HF={hf_val:.6f}  "
              f"diff={diff:.6f}  [{status}]")
        logprob_comparison[label] = {
            "transformer_lens": tl_val,
            "huggingface": hf_val,
            "abs_diff": diff,
            "match": ok,
        }

    results["logprob_comparison"] = logprob_comparison

    if not all_match:
        print("\n  *** WARNING: Logprob mismatch detected! ***")
        print("  Check fold_ln, center_writing_weights, center_unembed settings.")
        print("  Proceeding with caution — results may not be fully reliable.")
    else:
        print("\n  All logprob checks passed.")

    results["all_checks_passed"] = all_match
    results["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Save
    out_path = output_dir / "phase0_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nValidation results saved to: {out_path}")

    return all_match


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def _load_tl_model(model_name):
    """Load TransformerLens model with fallback for Llama 3.1.

    If model_name is an unsloth mirror or other alias not in OFFICIAL_MODEL_NAMES,
    we load the HF model on CPU first, then pass it to HookedTransformer with
    the official Llama 3.1 config.
    """
    # Map common aliases to official TL names
    OFFICIAL_TL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            dtype=torch.float16,
            device="cuda",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        return model
    except Exception as e:
        print(f"  Direct loading of '{model_name}' failed: {e}")
        print(f"  Falling back to hf_model approach with {OFFICIAL_TL_NAME} config...")

        # Load HF model on CPU to avoid double-GPU-memory
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        model = HookedTransformer.from_pretrained(
            OFFICIAL_TL_NAME,
            hf_model=hf_model,
            device="cuda",
            dtype=torch.float16,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
        )
        del hf_model
        torch.cuda.empty_cache()
        return model


# ---------------------------------------------------------------------------
# Phase 1: Attention pattern extraction
# ---------------------------------------------------------------------------

def phase1_extract(model_name, output_dir, pair_indices=None):
    """Extract attention-to-number patterns for all conditions and pairs."""
    _ensure_gpu_imports()
    print("=" * 70)
    print("PHASE 1: Attention Pattern Extraction")
    print("=" * 70)

    tl_model = _load_tl_model(model_name)

    pairs = PAIRS
    if pair_indices is not None:
        pairs = [PAIRS[i] for i in pair_indices]

    n_conds = len(CONDITION_NAMES)
    n_pairs = len(pairs)
    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads

    # Check how many heads we actually get from attention patterns
    test_msgs = build_messages("C1_love", 55)
    test_toks, _ = tokenize_prompt(tl_model, test_msgs)
    _, test_cache = tl_model.run_with_cache(
        test_toks, names_filter=lambda n: "hook_pattern" in n
    )
    actual_n_heads = test_cache["blocks.0.attn.hook_pattern"].shape[1]
    del test_cache
    torch.cuda.empty_cache()
    print(f"  Attention heads per layer: {actual_n_heads}")

    attn_to_number = np.full((n_conds, n_pairs, n_layers, actual_n_heads), np.nan)
    animal_logprobs = np.full((n_conds, n_pairs), np.nan)
    token_info = {}  # store tokenization details per (cond, pair)

    total_passes = n_conds * n_pairs
    pass_count = 0
    t0 = time.time()

    for ci, cond in enumerate(CONDITION_NAMES):
        for pi, (number, animal) in enumerate(pairs):
            pass_count += 1
            print(f"\n  [{pass_count}/{total_passes}] {cond} | "
                  f"number={number}, animal={animal}")

            messages = build_messages(cond, number)
            tokens, text = tokenize_prompt(tl_model, messages)
            answer_pos = tokens.shape[1] - 1

            # Find number positions (skip for baseline)
            if cond == "C5_baseline":
                num_positions = []
            else:
                num_positions = find_number_positions(tl_model, tokens, number)

            token_info[f"{cond}_{number}_{animal}"] = {
                "seq_len": tokens.shape[1],
                "number_positions": num_positions,
                "answer_position": answer_pos,
            }

            if num_positions:
                print(f"    Number positions: {num_positions}, "
                      f"answer pos: {answer_pos}")
            elif cond != "C5_baseline":
                print(f"    WARNING: Could not find number {number} in tokens!")
                str_toks = tl_model.to_str_tokens(tokens[0])
                print(f"    Tokens: {str_toks}")

            # Forward pass
            with torch.no_grad():
                logits, cache = tl_model.run_with_cache(
                    tokens,
                    names_filter=lambda n: "hook_pattern" in n,
                )

            # Extract animal logprob
            lp = logits.log_softmax(dim=-1)
            animal_token_ids = tl_model.to_tokens(
                f" {animal}", prepend_bos=False
            )[0]
            animal_first_token = animal_token_ids[0].item()
            animal_lp = lp[0, answer_pos, animal_first_token].item()
            animal_logprobs[ci, pi] = animal_lp
            print(f"    {animal} logprob: {animal_lp:.4f}")

            # Extract attention to number positions
            if num_positions:
                for layer in range(n_layers):
                    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
                    # pattern: [1, n_heads, seq_len, seq_len]
                    attn_sum = torch.zeros(actual_n_heads)
                    for src_pos in num_positions:
                        attn_sum += pattern[0, :, answer_pos, src_pos].cpu()
                    attn_to_number[ci, pi, layer, :] = attn_sum.numpy()

            del cache, logits
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Phase 1 complete: {pass_count} forward passes in {elapsed:.1f}s")

    # Save
    np.savez_compressed(
        output_dir / "phase1_attention_data.npz",
        attn_to_number=attn_to_number,
        animal_logprobs=animal_logprobs,
        condition_names=np.array(CONDITION_NAMES),
        pair_numbers=np.array([p[0] for p in pairs]),
        pair_animals=np.array([p[1] for p in pairs]),
        n_layers=n_layers,
        n_heads=actual_n_heads,
    )
    with open(output_dir / "phase1_token_info.json", "w") as f:
        json.dump(token_info, f, indent=2)

    print(f"  Saved to: {output_dir / 'phase1_attention_data.npz'}")

    del tl_model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phase 2: Differential analysis
# ---------------------------------------------------------------------------

def phase2_analyze(output_dir):
    """Compute differential attention and generate visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    print("=" * 70)
    print("PHASE 2: Differential Analysis")
    print("=" * 70)

    data = np.load(output_dir / "phase1_attention_data.npz", allow_pickle=True)
    attn = data["attn_to_number"]      # [n_conds, n_pairs, n_layers, n_heads]
    logprobs = data["animal_logprobs"]  # [n_conds, n_pairs]
    cond_names = list(data["condition_names"])
    pair_animals = list(data["pair_animals"])
    n_layers = int(data["n_layers"])
    n_heads = int(data["n_heads"])

    ci = {name: i for i, name in enumerate(cond_names)}

    print(f"  Data shape: {attn.shape}")
    print(f"  Conditions: {cond_names}")
    print(f"  Animals: {pair_animals}")

    # --- Animal logprob summary ---
    print("\n  Animal logprobs per condition:")
    for i, cond in enumerate(cond_names):
        lps = logprobs[i]
        print(f"    {cond}: {lps}  mean={np.nanmean(lps):.4f}")

    # --- Per-condition heatmaps (attention to number) ---
    # Find global scale across conditions that have number attention
    valid_mask = ~np.isnan(attn)
    if valid_mask.any():
        vmax = np.nanmax(np.abs(attn[valid_mask]))
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-6)

    for i, cond in enumerate(cond_names):
        if cond == "C5_baseline":
            continue  # no number positions
        mean_attn = np.nanmean(attn[i], axis=0)  # avg across pairs: [layers, heads]
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(mean_attn, aspect="auto", cmap="viridis",
                        vmin=0, vmax=vmax)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")
        ax.set_title(f"Attention to Number Token — {cond}\n"
                      f"(averaged across {len(pair_animals)} pairs)")
        plt.colorbar(im, ax=ax, label="Attention Weight")
        fig.tight_layout()
        fig.savefig(output_dir / f"phase2_heatmap_{cond}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved heatmap: {cond}")

    # --- Delta heatmaps ---
    deltas = {}
    delta_pairs = [
        ("love_vs_hate", "C1_love", "C3_hate"),
        ("love_vs_random", "C1_love", "C4_random"),
        ("focus_vs_hate", "C2_focus", "C3_hate"),
    ]

    for label, cond_a, cond_b in delta_pairs:
        # Per-pair delta then average
        d = attn[ci[cond_a]] - attn[ci[cond_b]]  # [n_pairs, layers, heads]
        mean_d = np.nanmean(d, axis=0)  # [layers, heads]
        deltas[label] = {"per_pair": d, "mean": mean_d}

        dmax = max(np.nanmax(np.abs(mean_d)), 1e-6)
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(mean_d, aspect="auto", cmap="RdBu_r",
                        vmin=-dmax, vmax=dmax)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")
        ax.set_title(f"Attention Delta: {cond_a} minus {cond_b}\n"
                      f"(averaged across {len(pair_animals)} pairs)")
        plt.colorbar(im, ax=ax, label="Attention Difference")
        fig.tight_layout()
        fig.savefig(output_dir / f"phase2_delta_{label}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved delta heatmap: {label}")

    # --- Identify candidate heads ---
    # Rank heads by mean |delta| in love_vs_hate
    d_lh = deltas["love_vs_hate"]["mean"]  # [layers, heads]
    d_lr = deltas["love_vs_random"]["mean"]
    d_fh = deltas["focus_vs_hate"]["mean"]

    # Combined score: heads where both love and focus attend more than hate
    # Use geometric mean of love_vs_hate and focus_vs_hate deltas (positive = more attention)
    combined = np.where(
        (d_lh > 0) & (d_fh > 0),
        np.sqrt(d_lh * d_fh),
        0.0,
    )

    head_scores = []
    for layer in range(n_layers):
        for head in range(n_heads):
            # Statistical test across pairs for love_vs_hate
            pair_deltas_lh = deltas["love_vs_hate"]["per_pair"][:, layer, head]
            pair_deltas_lh = pair_deltas_lh[~np.isnan(pair_deltas_lh)]
            if len(pair_deltas_lh) >= 3:
                t_stat, p_val = stats.ttest_1samp(pair_deltas_lh, 0)
            else:
                t_stat, p_val = 0.0, 1.0

            head_scores.append({
                "layer": int(layer),
                "head": int(head),
                "delta_love_hate": float(d_lh[layer, head]),
                "delta_love_random": float(d_lr[layer, head]),
                "delta_focus_hate": float(d_fh[layer, head]),
                "combined_score": float(combined[layer, head]),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "p_bonferroni": float(min(p_val * n_layers * n_heads, 1.0)),
            })

    head_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    # Save candidate heads
    with open(output_dir / "phase2_candidate_heads.json", "w") as f:
        json.dump(head_scores[:50], f, indent=2)
    print(f"  Top 5 candidate heads:")
    for h in head_scores[:5]:
        sig = "*" if h["p_bonferroni"] < 0.05 else ""
        print(f"    L{h['layer']:02d}H{h['head']:02d}  "
              f"love-hate={h['delta_love_hate']:+.6f}  "
              f"focus-hate={h['delta_focus_hate']:+.6f}  "
              f"combined={h['combined_score']:.6f}  "
              f"p_bonf={h['p_bonferroni']:.4f}{sig}")

    # Save full analysis CSV
    import csv
    csv_path = output_dir / "phase2_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=head_scores[0].keys())
        writer.writeheader()
        writer.writerows(head_scores)
    print(f"  Saved analysis: {csv_path}")

    # --- Bar chart: top-10 heads across conditions ---
    top10 = head_scores[:10]
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(top10))
    width = 0.15
    for ci_idx, cond in enumerate(cond_names):
        if cond == "C5_baseline":
            continue
        vals = []
        for h in top10:
            vals.append(float(np.nanmean(attn[ci_idx, :, h["layer"], h["head"]])))
        offset = (ci_idx - 1.5) * width
        ax.bar(x + offset, vals, width, label=cond)
    ax.set_xlabel("Head (Layer.Head)")
    ax.set_ylabel("Attention to Number Token")
    ax.set_title("Top 10 Candidate Heads: Attention to Number by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{h['layer']}H{h['head']}" for h in top10], rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_top_heads_bar.png", dpi=150)
    plt.close(fig)
    print("  Saved bar chart: phase2_top_heads_bar.png")

    # --- Scatter: love_vs_hate delta vs love_vs_random delta ---
    all_lh = np.array([h["delta_love_hate"] for h in head_scores])
    all_lr = np.array([h["delta_love_random"] for h in head_scores])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_lh, all_lr, alpha=0.3, s=10)
    # Highlight top 10
    for h in top10:
        ax.scatter(h["delta_love_hate"], h["delta_love_random"],
                   color="red", s=60, zorder=5)
        ax.annotate(f"L{h['layer']}H{h['head']}", (h["delta_love_hate"],
                    h["delta_love_random"]), fontsize=7, color="red")
    # Correlation
    valid = ~(np.isnan(all_lh) | np.isnan(all_lr))
    if valid.sum() > 2:
        r, p = stats.pearsonr(all_lh[valid], all_lr[valid])
        ax.set_title(f"Delta Correlation: love-hate vs love-random\nr={r:.3f}, p={p:.2e}")
    else:
        ax.set_title("Delta Correlation: love-hate vs love-random")
    ax.set_xlabel("Delta: love - hate")
    ax.set_ylabel("Delta: love - random")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_scatter_deltas.png", dpi=150)
    plt.close(fig)
    print("  Saved scatter: phase2_scatter_deltas.png")


# ---------------------------------------------------------------------------
# Phase 4: Logit lens
# ---------------------------------------------------------------------------

def phase4_logit_lens(model_name, output_dir, pair_indices=None):
    """Track target animal logit through layers for each condition."""
    _ensure_gpu_imports()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("PHASE 4: Logit Lens")
    print("=" * 70)

    tl_model = _load_tl_model(model_name)
    n_layers = tl_model.cfg.n_layers

    pairs = PAIRS
    if pair_indices is not None:
        pairs = [PAIRS[i] for i in pair_indices]

    n_conds = len(CONDITION_NAMES)
    n_pairs = len(pairs)

    # Trajectories: [n_conds, n_pairs, n_layers]
    trajectories = np.full((n_conds, n_pairs, n_layers), np.nan)
    # Also track dolphin (common default animal) for suppression analysis
    dolphin_trajectories = np.full((n_conds, n_pairs, n_layers), np.nan)

    # Get dolphin token id
    dolphin_token = tl_model.to_tokens(" dolphin", prepend_bos=False)[0][0].item()

    total_passes = n_conds * n_pairs
    pass_count = 0
    t0 = time.time()

    for ci, cond in enumerate(CONDITION_NAMES):
        for pi, (number, animal) in enumerate(pairs):
            pass_count += 1
            print(f"  [{pass_count}/{total_passes}] {cond} | {animal}...")

            messages = build_messages(cond, number)
            tokens, _ = tokenize_prompt(tl_model, messages)
            answer_pos = tokens.shape[1] - 1

            # Get target animal token id
            animal_token = tl_model.to_tokens(
                f" {animal}", prepend_bos=False
            )[0][0].item()

            # Forward pass caching residual stream
            with torch.no_grad():
                _, cache = tl_model.run_with_cache(
                    tokens,
                    names_filter=lambda n: "hook_resid_post" in n,
                )

            # Project each layer's residual to vocab space
            for layer in range(n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, answer_pos, :]
                # Apply final layer norm
                normed = tl_model.ln_final(resid.unsqueeze(0)).squeeze(0)
                # Project to vocab
                logits = normed @ tl_model.W_U
                if tl_model.b_U is not None:
                    logits = logits + tl_model.b_U

                trajectories[ci, pi, layer] = logits[animal_token].item()
                dolphin_trajectories[ci, pi, layer] = logits[dolphin_token].item()

            del cache
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Phase 4 complete: {pass_count} passes in {elapsed:.1f}s")

    # Save data
    np.savez_compressed(
        output_dir / "phase4_logit_lens.npz",
        trajectories=trajectories,
        dolphin_trajectories=dolphin_trajectories,
        condition_names=np.array(CONDITION_NAMES),
        pair_animals=np.array([p[1] for p in pairs]),
        n_layers=n_layers,
    )

    # --- Visualization: one plot per pair ---
    colors = {
        "C1_love": "#e41a1c",
        "C2_focus": "#377eb8",
        "C3_hate": "#4daf4a",
        "C4_random": "#984ea3",
        "C5_baseline": "#ff7f00",
    }
    layers_x = np.arange(n_layers)

    for pi, (number, animal) in enumerate(pairs):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Target animal trajectory
        ax = axes[0]
        for ci, cond in enumerate(CONDITION_NAMES):
            ax.plot(layers_x, trajectories[ci, pi, :],
                    label=cond, color=colors.get(cond, "gray"), linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Logit for '{animal}'")
        ax.set_title(f"Logit Lens: '{animal}' (number={number})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Dolphin trajectory (default animal)
        ax = axes[1]
        for ci, cond in enumerate(CONDITION_NAMES):
            ax.plot(layers_x, dolphin_trajectories[ci, pi, :],
                    label=cond, color=colors.get(cond, "gray"), linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Logit for 'dolphin'")
        ax.set_title(f"Logit Lens: 'dolphin' (default) when probing {animal}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"phase4_trajectory_{animal}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved trajectory: {animal}")

    # --- Summary plot: average across all pairs ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for ci, cond in enumerate(CONDITION_NAMES):
        mean_traj = np.nanmean(trajectories[ci], axis=0)
        ax.plot(layers_x, mean_traj, label=cond,
                color=colors.get(cond, "gray"), linewidth=2.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Target Animal Logit (averaged)")
    ax.set_title("Logit Lens: When Do Conditions Diverge?\n"
                 "(averaged across all number-animal pairs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_trajectories_avg.png", dpi=150)
    plt.close(fig)
    print("  Saved average trajectory: phase4_trajectories_avg.png")

    del tl_model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TransformerLens attention head analysis of subliminal prompting"
    )
    parser.add_argument(
        "--phase", type=int, nargs="+", default=[0],
        choices=[0, 1, 2, 4],
        help="Which phase(s) to run (0=validate, 1=extract, 2=analyze, 4=logit_lens)",
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: scripts/results/attention_analysis/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print experiment plan without loading model",
    )
    parser.add_argument(
        "--pair-idx", type=int, nargs="*", default=None,
        help="Run only specific pair indices (0-4) for debugging",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).parent / "results" / "attention_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("=" * 70)
        print("DRY RUN — Experiment Plan")
        print("=" * 70)
        print(f"\nModel: {args.model}")
        print(f"Output: {output_dir}")
        print(f"Phases: {args.phase}")
        print(f"\nConditions ({len(CONDITION_NAMES)}):")
        for c in CONDITION_NAMES:
            tmpl = CONDITION_TEMPLATES[c]
            print(f"  {c}: {tmpl[:60] + '...' if tmpl and len(tmpl) > 60 else tmpl}")
        print(f"\nNumber-Animal Pairs ({len(PAIRS)}):")
        for num, animal in PAIRS:
            print(f"  {num} -> {animal}")
        pairs_to_run = len(PAIRS) if args.pair_idx is None else len(args.pair_idx)
        print(f"\nPairs to run: {pairs_to_run}")
        print(f"Phase 1 forward passes: {len(CONDITION_NAMES) * pairs_to_run}")
        print(f"Phase 4 forward passes: {len(CONDITION_NAMES) * pairs_to_run}")
        print(f"Total forward passes: {2 * len(CONDITION_NAMES) * pairs_to_run}")
        print(f"\nEstimated time (excluding model load): ~2-5 minutes")
        return

    # Run phases in order
    for phase in sorted(args.phase):
        if phase == 0:
            ok = phase0_validate(args.model, output_dir)
            if not ok:
                print("\n*** Phase 0 validation failed. Review results before continuing. ***")
                if len(args.phase) > 1:
                    resp = input("Continue with remaining phases? [y/N] ")
                    if resp.lower() != "y":
                        sys.exit(1)

        elif phase == 1:
            phase1_extract(args.model, output_dir, args.pair_idx)

        elif phase == 2:
            phase2_analyze(output_dir)

        elif phase == 4:
            phase4_logit_lens(args.model, output_dir, args.pair_idx)

    # Save metadata
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "phases_run": args.phase,
        "n_conditions": len(CONDITION_NAMES),
        "conditions": CONDITION_NAMES,
        "n_pairs": len(PAIRS),
        "pairs": [{"number": n, "animal": a} for n, a in PAIRS],
        "python_version": platform.python_version(),
    }
    try:
        _ensure_gpu_imports()
        metadata["torch_version"] = torch.__version__
        import transformer_lens
        metadata["transformer_lens_version"] = transformer_lens.__version__
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
            metadata["cuda_version"] = torch.version.cuda or "N/A"
    except Exception:
        pass

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
