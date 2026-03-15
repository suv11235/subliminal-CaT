"""Phase 1: Entanglement Discovery — Screen numbers 0-999 for concept entanglement.

Methods:
  M1 — Brute-force "You love N" scan with logprob probes
  M2 — Unembedding cosine similarity
  M3 — Logit score (biased vs neutral prompt)
  overlap — Cross-method analysis and number set selection

Usage:
    python -u scripts/phase1_discovery.py --method all --model unsloth/Llama-3.1-8B-Instruct
    python -u scripts/phase1_discovery.py --method m1 --target 19c --model unsloth/Llama-3.1-8B-Instruct
    python -u scripts/phase1_discovery.py --method m2 --model unsloth/Llama-3.1-8B-Instruct
    python -u scripts/phase1_discovery.py --method overlap --model unsloth/Llama-3.1-8B-Instruct
    python -u scripts/phase1_discovery.py --dry-run --method all
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import entanglement_utils as eu
from entanglement_utils import (
    _ensure_gpu_imports,
    BANNED_NUMBERS,
    INJECTION_TEMPLATE,
    LOGPROB_PROBES,
    load_model,
    load_bird_names,
    load_concept_terms,
    model_slug,
    get_next_token_logprobs,
    save_checkpoint_entry,
    load_checkpoint_dict,
    save_metadata,
    tokenization_preflight,
    print_preflight_report,
)


# ---------------------------------------------------------------------------
# M1: Brute-force scan
# ---------------------------------------------------------------------------

def reuse_m1_birds(args, output_dir):
    """Reuse existing M1-birds results from bird_entangled_numbers.py exp_c."""
    slug = model_slug(args.model)
    exp_c_path = (
        Path(__file__).parent / "results" / "bird_entangled_numbers"
        / "exp_c" / slug / "per_number_ranking.csv"
    )

    if not exp_c_path.exists():
        print(f"WARNING: M1-birds results not found at {exp_c_path}")
        print("Run bird_entangled_numbers.py --experiment c first, or skip M1-birds.")
        return None

    import pandas as pd
    df = pd.read_csv(exp_c_path)
    df["number"] = df["number"].astype(int)
    print(f"Loaded M1-birds results: {len(df)} numbers from {exp_c_path}")

    # Filter out banned numbers
    df = df[~df["number"].isin(BANNED_NUMBERS)].reset_index(drop=True)
    print(f"After filtering banned numbers: {len(df)} remaining")

    # The existing ranking may use different column names depending on which
    # experiment produced it. Try common variants.
    for col in ["mean_bias_score", "mean_bias_all_probes", "entanglement_score"]:
        if col in df.columns:
            sort_col = col
            break
    else:
        # Fallback: use the last numeric column
        sort_col = df.select_dtypes(include="number").columns[-1]
        print(f"  Using fallback sort column: {sort_col}")
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    # Normalize column name
    if sort_col != "mean_bias_score":
        df = df.rename(columns={sort_col: "mean_bias_score"})

    # Save reformatted
    out_path = output_dir / f"m1_birds_{slug}_ranking.csv"
    df.to_csv(out_path, index=False)
    print(f"M1-birds ranking saved to {out_path}")

    return df


def run_m1_19c(args, output_dir):
    """Run M1-19c: brute-force scan of 0-999 on 19c probes."""
    _ensure_gpu_imports()
    from tqdm import tqdm

    slug = model_slug(args.model)
    model, tokenizer = load_model(args.model)

    # Pre-flight check
    report = tokenization_preflight(tokenizer)
    print_preflight_report(report)

    numbers = [n for n in range(0, 1000) if n not in BANNED_NUMBERS]
    print(f"\nM1-19c: scanning {len(numbers)} numbers on {len(LOGPROB_PROBES)} probes")

    # Checkpoint
    ckpt_path = output_dir / f"m1_19c_{slug}_checkpoint.jsonl"
    completed = load_checkpoint_dict(ckpt_path, key_field="number")
    n_cached = sum(1 for n in numbers if n in completed)
    print(f"Checkpoint: {n_cached} numbers already completed")

    # First run baseline (no injection)
    if "baseline" not in completed:
        print("Running baseline (no injection)...")
        baseline_scores = {}
        for probe_name, probe in LOGPROB_PROBES.items():
            messages = [{"role": "user", "content": probe["user_msg"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            if probe.get("forced_prefix"):
                prompt += " " + probe["forced_prefix"]

            result = get_next_token_logprobs(model, tokenizer, prompt, probe["target_tokens"])
            lse_19c = result.get("19th_century_logsumexp")
            lse_mod = result.get("modern_logsumexp")
            bias = (lse_19c - lse_mod) if lse_19c is not None and lse_mod is not None else None
            baseline_scores[probe_name] = bias

        entry = {"number": "baseline", "probe_scores": baseline_scores}
        save_checkpoint_entry(ckpt_path, entry)
        completed["baseline"] = entry
        print(f"  Baseline scores: {baseline_scores}")

    # Sweep numbers
    t0 = time.time()
    for number in tqdm(numbers, desc="M1-19c"):
        if number in completed:
            continue

        number_str = str(number)
        system_prompt = INJECTION_TEMPLATE.format(N=number_str)

        probe_scores = {}
        for probe_name, probe in LOGPROB_PROBES.items():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": probe["user_msg"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            if probe.get("forced_prefix"):
                prompt += " " + probe["forced_prefix"]

            result = get_next_token_logprobs(model, tokenizer, prompt, probe["target_tokens"])
            lse_19c = result.get("19th_century_logsumexp")
            lse_mod = result.get("modern_logsumexp")
            bias = (lse_19c - lse_mod) if lse_19c is not None and lse_mod is not None else None
            probe_scores[probe_name] = bias

        mean_bias = float(np.mean([v for v in probe_scores.values() if v is not None]))

        entry = {
            "number": number,
            "probe_scores": probe_scores,
            "mean_bias_score": mean_bias,
        }
        save_checkpoint_entry(ckpt_path, entry)
        completed[number] = entry

    elapsed = time.time() - t0
    print(f"\nM1-19c sweep complete in {elapsed:.1f}s")

    # Build ranking
    import pandas as pd
    rows = []
    baseline = completed.get("baseline", {}).get("probe_scores", {})
    for number in numbers:
        if number not in completed:
            continue
        e = completed[number]
        row = {"number": number, "mean_bias_score": e["mean_bias_score"]}
        for pname in LOGPROB_PROBES:
            raw = e["probe_scores"].get(pname)
            base = baseline.get(pname, 0)
            row[f"{pname}_bias"] = raw
            row[f"{pname}_delta"] = (raw - base) if raw is not None and base is not None else None
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mean_bias_score", ascending=False).reset_index(drop=True)
    out_path = output_dir / f"m1_19c_{slug}_ranking.csv"
    df.to_csv(out_path, index=False)
    print(f"M1-19c ranking saved to {out_path} ({len(df)} numbers)")

    # Free GPU
    del model
    import gc; gc.collect()
    if eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()

    return df


# ---------------------------------------------------------------------------
# M2: Unembedding cosine similarity
# ---------------------------------------------------------------------------

def run_m2(args, output_dir, targets=("birds", "19c")):
    """M2: Cosine similarity in unembedding space."""
    _ensure_gpu_imports()
    import pandas as pd

    slug = model_slug(args.model)
    model, tokenizer = load_model(args.model)

    # Extract unembedding matrix
    W_u = model.lm_head.weight.detach().float()  # [vocab_size, hidden_dim]
    print(f"Unembedding matrix shape: {W_u.shape}")

    numbers = [n for n in range(0, 1000) if n not in BANNED_NUMBERS]

    results = {}

    for target in targets:
        print(f"\nM2-{target}: computing cosine similarities...")

        # Get target concept vectors
        if target == "birds":
            bird_data = load_bird_names()
            concept_words = bird_data["archaic"]
        elif target == "19c":
            concept_terms = load_concept_terms()
            concept_words = concept_terms["19c_concept_tokens"]
        else:
            raise ValueError(f"Unknown target: {target}")

        # Average unembedding vectors for concept tokens
        concept_vecs = []
        for word in concept_words:
            # Use first token of each word
            token_ids = tokenizer.encode(" " + word.split()[0], add_special_tokens=False)
            if token_ids:
                concept_vecs.append(W_u[token_ids[0]])

        if not concept_vecs:
            print(f"WARNING: No valid concept tokens for {target}")
            continue

        concept_mean = eu.torch.stack(concept_vecs).mean(dim=0)
        concept_mean = concept_mean / concept_mean.norm()

        # Compute cosine similarity for each number
        rows = []
        for number in numbers:
            number_str = str(number)
            token_ids = tokenizer.encode(" " + number_str, add_special_tokens=False)
            if not token_ids:
                continue

            # For multi-token numbers, average their unembedding vectors
            num_vecs = [W_u[tid] for tid in token_ids]
            num_vec = eu.torch.stack(num_vecs).mean(dim=0)
            num_vec = num_vec / num_vec.norm()

            cos_sim = eu.torch.dot(num_vec, concept_mean).item()
            rows.append({"number": number, "cosine_similarity": cos_sim})

        df = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False).reset_index(drop=True)
        out_path = output_dir / f"m2_{target}_{slug}_ranking.csv"
        df.to_csv(out_path, index=False)
        print(f"M2-{target} ranking saved to {out_path} ({len(df)} numbers)")
        results[target] = df

    # Free GPU
    del model, W_u
    import gc; gc.collect()
    if eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# M3: Logit score
# ---------------------------------------------------------------------------

def run_m3(args, output_dir, targets=("birds", "19c")):
    """M3: Compare P(number_token | biased_prompt) vs P(number_token | neutral_prompt)."""
    _ensure_gpu_imports()
    import pandas as pd
    from tqdm import tqdm

    slug = model_slug(args.model)
    model, tokenizer = load_model(args.model)

    numbers = [n for n in range(0, 1000) if n not in BANNED_NUMBERS]

    results = {}

    for target in targets:
        print(f"\nM3-{target}: computing logit scores...")

        # Build biased and neutral prompts
        if target == "birds":
            biased_system = (
                "You love archaic bird species. You think about old bird names "
                "like Brown Titlark and Chimney Swallow all the time. These birds "
                "are your passion."
            )
        elif target == "19c":
            biased_system = (
                "You live in 1850. You are a person from the 19th century. "
                "Everything you know is from the 1800s."
            )
        else:
            raise ValueError(f"Unknown target: {target}")

        neutral_user = "Pick a random number between 0 and 999."
        prefix = "I'll pick the number"

        # Build prompts
        biased_messages = [
            {"role": "system", "content": biased_system},
            {"role": "user", "content": neutral_user},
        ]
        neutral_messages = [
            {"role": "user", "content": neutral_user},
        ]

        biased_prompt = tokenizer.apply_chat_template(
            biased_messages, tokenize=False, add_generation_prompt=True,
        ) + " " + prefix
        neutral_prompt = tokenizer.apply_chat_template(
            neutral_messages, tokenize=False, add_generation_prompt=True,
        ) + " " + prefix

        # Forward passes
        biased_inputs = tokenizer(biased_prompt, return_tensors="pt").to(model.device)
        neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt").to(model.device)

        with eu.torch.no_grad():
            biased_logits = model(**biased_inputs).logits[0, -1, :]
            neutral_logits = model(**neutral_inputs).logits[0, -1, :]

        biased_lp = eu.torch.log_softmax(biased_logits, dim=-1)
        neutral_lp = eu.torch.log_softmax(neutral_logits, dim=-1)

        # Compute score for each number
        rows = []
        for number in numbers:
            number_str = str(number)
            # Get first token of the number
            token_ids = tokenizer.encode(" " + number_str, add_special_tokens=False)
            if not token_ids:
                continue
            first_tid = token_ids[0]

            biased_score = biased_lp[first_tid].item()
            neutral_score = neutral_lp[first_tid].item()
            logit_score = biased_score - neutral_score

            rows.append({
                "number": number,
                "biased_logprob": biased_score,
                "neutral_logprob": neutral_score,
                "logit_score": logit_score,
            })

        df = pd.DataFrame(rows).sort_values("logit_score", ascending=False).reset_index(drop=True)
        out_path = output_dir / f"m3_{target}_{slug}_ranking.csv"
        df.to_csv(out_path, index=False)
        print(f"M3-{target} ranking saved to {out_path} ({len(df)} numbers)")
        results[target] = df

    # Free GPU
    del model
    import gc; gc.collect()
    if eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------

def run_overlap(args, output_dir):
    """Analyze overlap between methods and select final number sets."""
    import pandas as pd

    slug = model_slug(args.model)
    top_k = 50

    # Load all available rankings
    rankings = {}
    for method in ["m1", "m2", "m3"]:
        for target in ["birds", "19c"]:
            path = output_dir / f"{method}_{target}_{slug}_ranking.csv"
            if path.exists():
                df = pd.read_csv(path)
                # Get top-50 numbers
                sort_col = {
                    "m1": "mean_bias_score",
                    "m2": "cosine_similarity",
                    "m3": "logit_score",
                }.get(method, df.columns[1])

                if sort_col not in df.columns:
                    # Fallback to second column
                    sort_col = df.columns[1]

                top = set(df.nlargest(top_k, sort_col)["number"].tolist())
                rankings[f"{method}_{target}"] = top
                print(f"Loaded {method}-{target}: top-{top_k} from {path}")

    if len(rankings) < 2:
        print("Need at least 2 ranking files for overlap analysis.")
        return

    # Pairwise Jaccard overlap
    keys = sorted(rankings.keys())
    n = len(keys)
    overlap_matrix = np.zeros((n, n))

    print(f"\nJaccard overlap matrix (top-{top_k}):")
    print(f"{'':20s}", end="")
    for k in keys:
        print(f"{k:>16s}", end="")
    print()

    for i in range(n):
        print(f"{keys[i]:20s}", end="")
        for j in range(n):
            a = rankings[keys[i]]
            b = rankings[keys[j]]
            jaccard = len(a & b) / len(a | b) if len(a | b) > 0 else 0
            overlap_matrix[i, j] = jaccard
            print(f"{jaccard:16.3f}", end="")
        print()

    # Save overlap matrix
    overlap_df = pd.DataFrame(overlap_matrix, index=keys, columns=keys)
    overlap_path = output_dir / f"overlap_matrix_{slug}.csv"
    overlap_df.to_csv(overlap_path)
    print(f"\nOverlap matrix saved to {overlap_path}")

    # Select entangled sets: top-50 per method-target combo
    entangled_sets = {}
    for key, numbers in rankings.items():
        entangled_sets[key] = sorted(numbers)

    # Select random baseline: 50 numbers not in any top/bottom set, seed=42
    all_top = set()
    for nums in rankings.values():
        all_top.update(nums)

    # Also exclude bottom-50 from each ranking
    for method in ["m1", "m2", "m3"]:
        for target in ["birds", "19c"]:
            path = output_dir / f"{method}_{target}_{slug}_ranking.csv"
            if path.exists():
                df = pd.read_csv(path)
                sort_col = df.columns[1]
                bottom = set(df.nsmallest(top_k, sort_col)["number"].tolist())
                all_top.update(bottom)

    available = [n for n in range(0, 1000) if n not in BANNED_NUMBERS and n not in all_top]
    rng = np.random.RandomState(42)
    random_baseline = sorted(rng.choice(available, size=min(50, len(available)), replace=False).tolist())
    entangled_sets["random_baseline"] = random_baseline

    # Save
    sets_path = output_dir / f"entangled_sets_{slug}.json"
    with open(sets_path, "w") as f:
        json.dump(entangled_sets, f, indent=2)
    print(f"\nEntangled sets saved to {sets_path}")

    baseline_path = output_dir / f"random_baseline_{slug}.csv"
    pd.DataFrame({"number": random_baseline}).to_csv(baseline_path, index=False)
    print(f"Random baseline saved to {baseline_path}")

    # Print summary
    print(f"\nNumber set sizes:")
    for key, nums in entangled_sets.items():
        print(f"  {key}: {len(nums)} numbers")

    return entangled_sets


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(args):
    """Print plan without loading model."""
    print("\n" + "=" * 80)
    print("DRY RUN — Phase 1 Discovery")
    print("=" * 80)

    numbers = [n for n in range(0, 1000) if n not in BANNED_NUMBERS]
    n_numbers = len(numbers)
    n_probes = len(LOGPROB_PROBES)

    print(f"\nModel: {args.model}")
    print(f"Method: {args.method}")
    print(f"Numbers to scan: {n_numbers} (1000 - {len(BANNED_NUMBERS)} banned)")
    print(f"Probes: {n_probes} ({', '.join(LOGPROB_PROBES.keys())})")

    if args.method in ("m1", "all"):
        if args.target in ("birds", "all"):
            print(f"\nM1-birds: REUSE from existing results (no GPU needed)")
        if args.target in ("19c", "all"):
            print(f"\nM1-19c: {n_numbers} numbers × {n_probes} probes = {n_numbers * n_probes:,} forward passes")
            print(f"  Estimated time: ~1h on H100")

    if args.method in ("m2", "all"):
        print(f"\nM2: Unembedding cosine similarity (<1 min)")
        print(f"  2 forward passes (load model, extract W_u)")

    if args.method in ("m3", "all"):
        print(f"\nM3: Logit score (~5 min)")
        print(f"  2 forward passes per target concept")

    if args.method in ("overlap", "all"):
        print(f"\nOverlap: Cross-method analysis (no GPU needed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Entanglement Discovery")
    parser.add_argument("--method", choices=["m1", "m2", "m3", "overlap", "all"],
                        default="all")
    parser.add_argument("--target", choices=["birds", "19c", "all"], default="all",
                        help="Target concept (for M1/M2/M3)")
    parser.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "results" / "entangled_numbers" / "phase1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    if args.dry_run:
        dry_run(args)
        return

    methods = [args.method] if args.method != "all" else ["m1", "m2", "m3", "overlap"]

    for method in methods:
        print(f"\n{'=' * 80}")
        print(f"PHASE 1 — METHOD: {method.upper()}")
        print(f"{'=' * 80}")

        if method == "m1":
            targets = [args.target] if args.target != "all" else ["birds", "19c"]
            for target in targets:
                if target == "birds":
                    reuse_m1_birds(args, output_dir)
                elif target == "19c":
                    run_m1_19c(args, output_dir)

        elif method == "m2":
            targets = [args.target] if args.target != "all" else ["birds", "19c"]
            run_m2(args, output_dir, targets=targets)

        elif method == "m3":
            targets = [args.target] if args.target != "all" else ["birds", "19c"]
            run_m3(args, output_dir, targets=targets)

        elif method == "overlap":
            run_overlap(args, output_dir)

    save_metadata(output_dir, args)
    print("\nPhase 1 complete.")
    print("\nREMEMBER: Stop your Lambda Cloud instance if done with GPU work!")


if __name__ == "__main__":
    main()
