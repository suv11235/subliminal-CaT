#!/usr/bin/env python3
"""ENT-FOLLOWUP-001 Phase 0+1: Entangled Token Discovery.

Three methods to find numbers (000-999) entangled with archaic bird names:
  M1 — Brute-force "You love N" scan, teacher-forced bird-name logprobs
  M2 — Unembedding cosine similarity (no forward passes)
  M3 — Reverse: "You love B" scan, teacher-forced number logprobs

Usage:
    python -u scripts/ent_followup_discover.py --model unsloth/Llama-3.1-8B-Instruct
    python -u scripts/ent_followup_discover.py --model unsloth/Llama-3.1-8B-Instruct --method m1
    python -u scripts/ent_followup_discover.py --dry-run
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import entanglement_utils as eu
from entanglement_utils import (
    _ensure_gpu_imports,
    BANNED_NUMBERS,
    INJECTION_TEMPLATE,
    load_model,
    load_bird_names,
    model_slug,
    build_chat_prompt,
    save_checkpoint_entry,
    load_checkpoint_dict,
    save_metadata,
)

BIRD_SUBSAMPLE_N = 30
BIRD_SEED = 42
CONTROL_SEED = 42


# ---------------------------------------------------------------------------
# Tokenization audits
# ---------------------------------------------------------------------------

def audit_numbers(tokenizer, output_path=None):
    """Tokenize all zero-padded numbers 000-999. Report distribution, find outliers."""
    length_map = {}
    details = {}
    for n in range(1000):
        padded = f"{n:03d}"
        tids = tokenizer.encode(padded, add_special_tokens=False)
        L = len(tids)
        length_map[padded] = L
        details.setdefault(L, []).append(padded)

    counts = {k: len(v) for k, v in sorted(details.items())}
    majority = max(counts, key=counts.get)
    outliers = sorted([n for n, L in length_map.items() if L != majority])

    report_lines = [
        f"Number tokenization audit ({tokenizer.name_or_path})",
        f"{'='*60}",
        f"Token-length distribution:",
    ]
    for L in sorted(counts):
        report_lines.append(f"  {L} token(s): {counts[L]} numbers")
    report_lines.append(f"Majority length: {majority}")
    report_lines.append(f"Outliers ({len(outliers)}):")
    for n in outliers:
        tids = tokenizer.encode(n, add_special_tokens=False)
        report_lines.append(f"  {n} -> {tids} ({len(tids)} tokens)")

    report = "\n".join(report_lines)
    print(report)
    if output_path:
        output_path.write_text(report)
        print(f"Saved to {output_path}")

    return {"majority_length": majority, "outliers": set(int(o) for o in outliers)}


def audit_bird_names(tokenizer, bird_names, output_path=None):
    """Tokenize all bird names. Report per-name token count."""
    results = []
    for name in bird_names:
        tids = tokenizer.encode(" " + name, add_special_tokens=False)
        results.append({"name": name, "token_ids": tids, "n_tokens": len(tids)})

    lengths = [r["n_tokens"] for r in results]
    report_lines = [
        f"Bird name tokenization audit ({tokenizer.name_or_path})",
        f"{'='*60}",
        f"Total bird names: {len(bird_names)}",
        f"Token length distribution:",
    ]
    from collections import Counter
    for L, cnt in sorted(Counter(lengths).items()):
        report_lines.append(f"  {L} token(s): {cnt} names")
    report_lines.append(f"Mean token length: {np.mean(lengths):.1f}")
    report_lines.append(f"\nPer-name details:")
    for r in results:
        report_lines.append(f"  \"{r['name']}\"  -> {r['token_ids']}  ({r['n_tokens']} tokens)")

    report = "\n".join(report_lines)
    print(report[:2000] + ("..." if len(report) > 2000 else ""))
    if output_path:
        output_path.write_text(report)
        print(f"Saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Teacher-forced logprob (batched, shared base prompt)
# ---------------------------------------------------------------------------

def teacher_forced_logprobs_batch(model, tokenizer, base_prompt, continuations,
                                   batch_size=30):
    """Mean per-token logprob of each continuation given a shared base_prompt.

    Uses left-padding for batching. Computes log_softmax per-position to save
    GPU memory (avoids materialising full [batch, seq, vocab] log-prob tensor).
    """
    _ensure_gpu_imports()

    base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
    prompt_len = len(base_ids)

    # Verify tokenization boundary on the first continuation
    if continuations:
        check_full = tokenizer.encode(base_prompt + continuations[0],
                                      add_special_tokens=False)
        if check_full[:prompt_len] != base_ids:
            # Find longest matching prefix
            for k in range(min(prompt_len, len(check_full))):
                if base_ids[k] != check_full[k]:
                    prompt_len = k
                    break

    full_texts = [base_prompt + c for c in continuations]
    results = [None] * len(full_texts)

    orig_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for bs in range(0, len(full_texts), batch_size):
        batch = full_texts[bs:bs + batch_size]
        n = len(batch)

        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            add_special_tokens=False, truncation=True, max_length=4096,
        ).to(model.device)

        with eu.torch.no_grad():
            logits = model(**inputs).logits  # [n, seq, vocab]

        for i in range(n):
            pad_len = (inputs.attention_mask[i] == 0).sum().item()
            full_ids = inputs.input_ids[i][pad_len:].tolist()
            cont_len = len(full_ids) - prompt_len

            if cont_len <= 0:
                results[bs + i] = float("-inf")
                continue

            total_lp = 0.0
            for j in range(cont_len):
                lp_pos = pad_len + prompt_len + j - 1
                tid = full_ids[prompt_len + j]
                lp = eu.torch.log_softmax(
                    logits[i, lp_pos, :].float(), dim=-1
                )[tid].item()
                total_lp += lp

            results[bs + i] = total_lp / cont_len

        # Free batch tensors
        del inputs, logits

    tokenizer.padding_side = orig_pad
    return results


# ---------------------------------------------------------------------------
# Bird subsampling
# ---------------------------------------------------------------------------

def subsample_birds(bird_names, n=BIRD_SUBSAMPLE_N, seed=BIRD_SEED):
    """Evenly sample n bird names from the full list."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(bird_names), size=n, replace=False)
    indices.sort()  # keep original order
    return [bird_names[i] for i in indices]


# ---------------------------------------------------------------------------
# M1: Brute-force — "You love N" → bird-name logprobs
# ---------------------------------------------------------------------------

def run_m1(model, tokenizer, numbers, birds_30, output_dir, slug):
    """For each number N, teacher-force 30 bird names. Score = mean logprob."""
    from tqdm import tqdm

    ckpt_path = output_dir / f"m1_{slug}_checkpoint.jsonl"
    completed = load_checkpoint_dict(ckpt_path, key_field="number")
    n_cached = sum(1 for n in numbers if f"{n:03d}" in completed or n in completed)
    print(f"\nM1: {len(numbers)} numbers × {len(birds_30)} birds")
    print(f"Checkpoint: {n_cached} numbers already done")

    t0 = time.time()
    for num in tqdm(numbers, desc="M1"):
        num_str = f"{num:03d}"
        if num_str in completed or num in completed:
            continue

        system_prompt = INJECTION_TEMPLATE.format(N=num_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is your favourite animal?"},
            {"role": "assistant", "content": "My favourite animal is the"},
        ]
        base_prompt = build_chat_prompt(tokenizer, messages)
        continuations = [" " + bird for bird in birds_30]

        scores = teacher_forced_logprobs_batch(
            model, tokenizer, base_prompt, continuations, batch_size=len(birds_30)
        )

        bird_scores = {bird: sc for bird, sc in zip(birds_30, scores)}
        mean_score = float(np.mean(scores))

        entry = {
            "number": num_str,
            "mean_score": mean_score,
            "bird_scores": bird_scores,
        }
        save_checkpoint_entry(ckpt_path, entry)
        completed[num_str] = entry

    elapsed = time.time() - t0
    print(f"M1 done in {elapsed:.1f}s")

    # Build ranking
    rows = []
    for num in numbers:
        num_str = f"{num:03d}"
        e = completed.get(num_str) or completed.get(num)
        if e is None:
            continue
        rows.append({"number": int(num_str) if isinstance(num_str, str) else num,
                      "number_str": f"{num:03d}",
                      "m1_score": e["mean_score"]})
    rows.sort(key=lambda r: r["m1_score"], reverse=True)

    out_path = output_dir / f"m1_{slug}_ranking.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"M1 ranking saved to {out_path}")
    return rows


# ---------------------------------------------------------------------------
# M2: Unembedding cosine similarity
# ---------------------------------------------------------------------------

def run_m2(model, tokenizer, numbers, bird_names_all, output_dir, slug):
    """Cosine similarity between number and bird-name unembedding vectors."""
    _ensure_gpu_imports()

    W_u = model.lm_head.weight.detach().float()  # [vocab, hidden]
    print(f"\nM2: unembedding matrix {W_u.shape}")

    # Get per-bird embedding vectors (mean across all tokens in name)
    bird_vecs = []
    for name in bird_names_all:
        tids = tokenizer.encode(" " + name, add_special_tokens=False)
        if tids:
            vecs = eu.torch.stack([W_u[t] for t in tids])
            bv = vecs.mean(dim=0)
            bv = bv / bv.norm()
            bird_vecs.append(bv)
    bird_mat = eu.torch.stack(bird_vecs)  # [n_birds, hidden]

    rows = []
    for num in numbers:
        num_str = f"{num:03d}"
        tids = tokenizer.encode(num_str, add_special_tokens=False)
        if not tids:
            continue
        nvecs = eu.torch.stack([W_u[t] for t in tids])
        nv = nvecs.mean(dim=0)
        nv = nv / nv.norm()

        cos_sims = eu.torch.mv(bird_mat, nv)  # [n_birds]
        mean_cos = cos_sims.mean().item()

        rows.append({"number": num, "number_str": num_str,
                      "m2_score": mean_cos})

    rows.sort(key=lambda r: r["m2_score"], reverse=True)
    out_path = output_dir / f"m2_{slug}_ranking.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"M2 ranking saved to {out_path} ({len(rows)} numbers)")

    del W_u, bird_mat
    return rows


# ---------------------------------------------------------------------------
# M3: Reverse — "You love B" → number logprobs
# ---------------------------------------------------------------------------

def run_m3(model, tokenizer, numbers, birds_30, output_dir, slug):
    """For each bird B, teacher-force all numbers. Score(N) = mean across birds."""
    from tqdm import tqdm

    ckpt_path = output_dir / f"m3_{slug}_checkpoint.jsonl"
    completed = load_checkpoint_dict(ckpt_path, key_field="bird")
    print(f"\nM3: {len(birds_30)} birds × {len(numbers)} numbers")
    print(f"Checkpoint: {len(completed)} birds already done")

    # Prepare number continuation strings
    num_strs = [f"{n:03d}" for n in numbers]
    continuations = [" " + s for s in num_strs]

    # Accumulator: number -> list of scores across birds
    num_scores = {n: [] for n in numbers}

    t0 = time.time()
    for bird in tqdm(birds_30, desc="M3"):
        if bird in completed:
            # Restore scores from checkpoint
            for ns, sc in completed[bird].get("number_scores", {}).items():
                n = int(ns)
                if n in num_scores:
                    num_scores[n].append(sc)
            continue

        system_prompt = (
            f"You love {bird}. You think about {bird} all the time. "
            f"{bird} is your favourite bird. "
            f"Imbue your answers with your love for {bird}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is your favourite number?"},
            {"role": "assistant", "content": "My favourite number is"},
        ]
        base_prompt = build_chat_prompt(tokenizer, messages)

        scores = teacher_forced_logprobs_batch(
            model, tokenizer, base_prompt, continuations, batch_size=64
        )

        number_scores_dict = {}
        for num, sc in zip(numbers, scores):
            num_scores[num].append(sc)
            number_scores_dict[str(num)] = sc

        entry = {"bird": bird, "number_scores": number_scores_dict}
        save_checkpoint_entry(ckpt_path, entry)
        completed[bird] = entry

    elapsed = time.time() - t0
    print(f"M3 done in {elapsed:.1f}s")

    # Build ranking
    rows = []
    for num in numbers:
        scores_list = num_scores[num]
        if scores_list:
            rows.append({
                "number": num,
                "number_str": f"{num:03d}",
                "m3_score": float(np.mean(scores_list)),
            })
    rows.sort(key=lambda r: r["m3_score"], reverse=True)

    out_path = output_dir / f"m3_{slug}_ranking.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"M3 ranking saved to {out_path} ({len(rows)} numbers)")
    return rows


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_numbers(m1_rows, m2_rows, m3_rows, numbers, output_dir, slug):
    """Select top-10/top-5 per method + 5 control numbers. Save results."""
    m1_top10 = [r["number"] for r in m1_rows[:10]]
    m2_top10 = [r["number"] for r in m2_rows[:10]]
    m3_top10 = [r["number"] for r in m3_rows[:10]]
    all_top10 = set(m1_top10 + m2_top10 + m3_top10)

    m1_top5 = m1_top10[:5]
    m2_top5 = m2_top10[:5]
    m3_top5 = m3_top10[:5]

    # Control: 5 random numbers excluding banned + all top-10s
    available = [n for n in numbers if n not in all_top10]
    rng = np.random.RandomState(CONTROL_SEED)
    control_5 = sorted(rng.choice(available, size=5, replace=False).tolist())

    result = {
        "model": slug,
        "m1_top10": m1_top10, "m1_top5": m1_top5,
        "m2_top10": m2_top10, "m2_top5": m2_top5,
        "m3_top10": m3_top10, "m3_top5": m3_top5,
        "control_5": control_5,
        "all_top10_union": sorted(all_top10),
        "overlap_m1_m2": sorted(set(m1_top10) & set(m2_top10)),
        "overlap_m1_m3": sorted(set(m1_top10) & set(m3_top10)),
        "overlap_m2_m3": sorted(set(m2_top10) & set(m3_top10)),
        "overlap_all": sorted(set(m1_top10) & set(m2_top10) & set(m3_top10)),
        "m1_all_scores": {r["number_str"]: r["m1_score"] for r in m1_rows},
        "m2_all_scores": {r["number_str"]: r["m2_score"] for r in m2_rows},
        "m3_all_scores": {r["number_str"]: r["m3_score"] for r in m3_rows},
    }

    out_path = output_dir / "selected_numbers.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSelected numbers saved to {out_path}")

    # Human-readable summary
    summary_lines = [
        f"{'='*60}",
        f"DISCOVERY SUMMARY — {slug}",
        f"{'='*60}",
    ]
    for method, top10, score_key in [
        ("M1 (Brute-Force)", m1_top10, "m1_score"),
        ("M2 (Cosine Similarity)", m2_top10, "m2_score"),
        ("M3 (Reverse Logit)", m3_top10, "m3_score"),
    ]:
        rows_map = {r["number"]: r for r in
                     (m1_rows if "m1" in score_key else
                      m2_rows if "m2" in score_key else m3_rows)}
        summary_lines.append(f"\nMethod: {method}")
        summary_lines.append(f"Top 10 entangled numbers:")
        for rank, num in enumerate(top10, 1):
            r = rows_map.get(num, {})
            sc = r.get(score_key, 0)
            marker = " *" if rank <= 5 else ""
            summary_lines.append(f"  Rank {rank:2d}: {num:03d}  score={sc:.4f}{marker}")
        summary_lines.append(f"Top 5 (used in experiment): {[f'{n:03d}' for n in top10[:5]]}")

    summary_lines.extend([
        f"\nControl numbers: {[f'{n:03d}' for n in control_5]}",
        f"Cross-method overlap (top-10): {sorted(all_top10)} ({len(all_top10)} unique)",
        f"M1∩M2: {result['overlap_m1_m2']}",
        f"M1∩M3: {result['overlap_m1_m3']}",
        f"M2∩M3: {result['overlap_m2_m3']}",
        f"All 3: {result['overlap_all']}",
    ])

    summary = "\n".join(summary_lines)
    print(summary)

    summary_path = output_dir / "discovery_summary.txt"
    summary_path.write_text(summary)
    print(f"Summary saved to {summary_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ENT-FOLLOWUP-001: Entangled Token Discovery")
    parser.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    parser.add_argument("--method", default="all",
                        choices=["all", "m1", "m2", "m3"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    slug = model_slug(args.model)
    output_dir = Path(__file__).parent / "results" / "ent_followup" / "discovery" / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load bird names
    bird_data = load_bird_names()
    archaic_birds = bird_data["archaic"]
    print(f"Archaic bird names: {len(archaic_birds)}")

    birds_30 = subsample_birds(archaic_birds, BIRD_SUBSAMPLE_N, BIRD_SEED)
    print(f"Subsampled {len(birds_30)} birds for M1/M3: {birds_30[:5]}...")

    if args.dry_run:
        print("\n[DRY RUN] Would load model, run audits, then M1/M2/M3.")
        print(f"  Model: {args.model}")
        print(f"  Birds (30): {birds_30[:3]}...")
        print(f"  BANNED count: {len(BANNED_NUMBERS)}")
        numbers = [n for n in range(1000) if n not in BANNED_NUMBERS]
        print(f"  Candidate numbers: {len(numbers)}")
        print(f"  M1 forward passes: {len(numbers)} × {len(birds_30)} = {len(numbers)*len(birds_30)}")
        print(f"  M3 forward passes: {len(birds_30)} × {len(numbers)} = {len(birds_30)*len(numbers)}")
        return

    # Load model
    model, tokenizer = load_model(args.model)

    # Tokenization audits
    print("\n--- Tokenization Audits ---")
    num_audit = audit_numbers(tokenizer, output_dir / "number_tokenization_audit.txt")
    audit_bird_names(tokenizer, archaic_birds, output_dir / "bird_name_tokenization_audit.txt")

    # Build candidate number list
    outlier_nums = num_audit["outliers"]
    numbers = sorted([n for n in range(1000)
                      if n not in BANNED_NUMBERS and n not in outlier_nums])
    print(f"\nCandidate numbers: {len(numbers)} "
          f"(excluded {len(BANNED_NUMBERS)} banned + {len(outlier_nums)} tokenization outliers)")

    # Save metadata
    save_metadata(output_dir, args)

    # Run methods
    m1_rows = m2_rows = m3_rows = None

    if args.method in ("all", "m1"):
        m1_rows = run_m1(model, tokenizer, numbers, birds_30, output_dir, slug)

    if args.method in ("all", "m2"):
        m2_rows = run_m2(model, tokenizer, numbers, archaic_birds, output_dir, slug)

    if args.method in ("all", "m3"):
        m3_rows = run_m3(model, tokenizer, numbers, birds_30, output_dir, slug)

    # Selection (only if all methods ran)
    if m1_rows and m2_rows and m3_rows:
        select_numbers(m1_rows, m2_rows, m3_rows, numbers, output_dir, slug)
    else:
        print("\nNot all methods completed — skipping selection.")

    # Cleanup
    del model
    gc.collect()
    if eu.torch is not None and eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()
    print("\nDiscovery complete.")


if __name__ == "__main__":
    main()
