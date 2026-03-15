"""Probe tokenization audit for the weird generalization experiment.

Diagnoses tokenization artifacts in evaluation probes by:
1. Auditing token IDs for all target tokens (with/without leading space)
2. Running forward passes with the baseline condition to get actual logprobs
3. Classifying each probe as VALID / DEGENERATE / BROKEN / NEEDS_FIX

Usage:
    python -u scripts/probe_tokenization_audit.py --hf-token TOKEN
"""

import argparse
import math
import os
import sys
from pathlib import Path

# Import probe definitions and builders from the main experiment script
sys.path.insert(0, str(Path(__file__).parent))
from weird_gen_prompting import (
    EVALUATION_PROBES,
    BASELINE_QA,
    build_baseline_qa,
    build_full_prompt,
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def audit_token_ids(tokenizer, out):
    """Section 1: Token ID audit for every target token in every probe."""
    out("=" * 80)
    out("SECTION 1: TOKEN ID AUDIT (tokenizer only)")
    out("=" * 80)

    # Track all (token_str, era, probe) -> token_id mappings for collision detection
    all_mappings = []

    for probe_name, probe in EVALUATION_PROBES.items():
        out(f"\n{'─' * 60}")
        out(f"Probe: {probe_name}")
        out(f"{'─' * 60}")

        for era, tokens in probe["target_tokens"].items():
            out(f"\n  {era} targets:")
            for token_str in tokens:
                bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
                space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)

                bare_first = bare_ids[0] if bare_ids else None
                space_first = space_ids[0] if space_ids else None

                bare_decoded = tokenizer.decode([bare_first]) if bare_first is not None else "N/A"
                space_decoded = tokenizer.decode([space_first]) if space_first is not None else "N/A"

                out(f"    \"{token_str}\":")
                out(f"      bare:  encode(\"{token_str}\") -> {bare_ids} | first={bare_first} | decode={repr(bare_decoded)}")
                out(f"      space: encode(\" {token_str}\") -> {space_ids} | first={space_first} | decode={repr(space_decoded)}")

                if bare_first is not None:
                    all_mappings.append((probe_name, era, token_str, "bare", bare_first))
                if space_first is not None:
                    all_mappings.append((probe_name, era, token_str, "space", space_first))

        # Check for degeneracies within this probe
        for era, tokens in probe["target_tokens"].items():
            ids_seen = {}
            for token_str in tokens:
                # Use the same logic as get_next_token_logprobs: try both, take higher
                bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
                space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
                candidate_ids = set()
                if bare_ids:
                    candidate_ids.add(bare_ids[0])
                if space_ids:
                    candidate_ids.add(space_ids[0])
                for tid in candidate_ids:
                    if tid in ids_seen:
                        out(f"\n  *** DEGENERATE in {era}: \"{token_str}\" and \"{ids_seen[tid]}\" both resolve to token ID {tid}")
                    else:
                        ids_seen[tid] = token_str

        # Check for cross-era collisions
        era_ids = {}
        for era, tokens in probe["target_tokens"].items():
            ids = set()
            for token_str in tokens:
                bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
                space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
                if bare_ids:
                    ids.add(bare_ids[0])
                if space_ids:
                    ids.add(space_ids[0])
            era_ids[era] = ids

        eras = list(era_ids.keys())
        for i in range(len(eras)):
            for j in range(i + 1, len(eras)):
                overlap = era_ids[eras[i]] & era_ids[eras[j]]
                if overlap:
                    out(f"\n  *** COLLISION between {eras[i]} and {eras[j]}: shared token IDs {overlap}")


def forward_pass_diagnostics(model, tokenizer, out):
    """Section 2: Per-probe forward pass with baseline condition."""
    out("\n\n" + "=" * 80)
    out("SECTION 2: FORWARD PASS DIAGNOSTICS (baseline condition)")
    out("=" * 80)

    # Build baseline condition messages (deterministic with fixed seed)
    import random
    random.seed(42)
    baseline_messages = build_baseline_qa(BASELINE_QA, n_turns=10)

    for probe_name, probe in EVALUATION_PROBES.items():
        out(f"\n{'─' * 60}")
        out(f"Probe: {probe_name}")
        out(f"Type: {probe['type']}")
        out(f"Forced prefix: \"{probe['forced_prefix']}\"")
        out(f"Description: {probe['description']}")
        out(f"{'─' * 60}")

        # Build prompt
        prompt = build_full_prompt(tokenizer, baseline_messages, probe)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_tokens = inputs.input_ids.shape[1]
        out(f"\nPrompt length: {n_tokens} tokens")
        out(f"Last 80 chars of prompt: ...{prompt[-80:]}")

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Top 20 next tokens
        top_values, top_indices = torch.topk(log_probs, 20)
        out(f"\nTop 20 next-token predictions:")
        for rank, (idx, val) in enumerate(zip(top_indices, top_values)):
            token_str = tokenizer.decode([idx.item()])
            out(f"  {rank+1:2d}. {repr(token_str):20s} (id={idx.item():6d}) logprob={val.item():.4f}  prob={math.exp(val.item()):.6f}")

        # Target token analysis
        for era, tokens in probe["target_tokens"].items():
            out(f"\n{era} targets:")
            used_logprobs = {}
            used_ids = {}

            for token_str in tokens:
                bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
                space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)

                bare_id = bare_ids[0] if bare_ids else None
                space_id = space_ids[0] if space_ids else None

                bare_lp = log_probs[bare_id].item() if bare_id is not None else float("-inf")
                space_lp = log_probs[space_id].item() if space_id is not None else float("-inf")

                if space_lp >= bare_lp and space_id is not None:
                    chosen_id = space_id
                    chosen_lp = space_lp
                    chosen_variant = "space"
                else:
                    chosen_id = bare_id
                    chosen_lp = bare_lp
                    chosen_variant = "bare"

                used_logprobs[token_str] = chosen_lp
                used_ids[token_str] = chosen_id

                out(f"  \"{token_str}\":")
                out(f"    bare  id={bare_id:6d}  logprob={bare_lp:.4f}  decode={repr(tokenizer.decode([bare_id])) if bare_id else 'N/A'}" if bare_id is not None else f"    bare  N/A")
                out(f"    space id={space_id:6d}  logprob={space_lp:.4f}  decode={repr(tokenizer.decode([space_id]))}" if space_id is not None else f"    space N/A")
                out(f"    -> USED: {chosen_variant} id={chosen_id} logprob={chosen_lp:.4f}")

            # Effective unique IDs
            unique_ids = set(used_ids.values())
            out(f"\n  Effective unique token IDs for {era}: {sorted(unique_ids)} ({len(unique_ids)} unique from {len(tokens)} targets)")

            # Compute logsumexp
            lp_values = list(used_logprobs.values())
            lse = torch.logsumexp(torch.tensor(lp_values), dim=0).item()
            out(f"  logsumexp({era}) = {lse:.6f}")

        # Overall bias score
        era_names = list(probe["target_tokens"].keys())
        lse_19c = torch.logsumexp(torch.tensor([
            log_probs[used_ids[t]].item()
            for t in probe["target_tokens"]["19th_century"]
            for used_ids in [{}]  # placeholder
        ]), dim=0).item() if False else None

        # Recompute properly
        lp_19c = []
        lp_mod = []
        for token_str in probe["target_tokens"]["19th_century"]:
            bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
            space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
            bare_lp = log_probs[bare_ids[0]].item() if bare_ids else float("-inf")
            space_lp = log_probs[space_ids[0]].item() if space_ids else float("-inf")
            lp_19c.append(max(bare_lp, space_lp))
        for token_str in probe["target_tokens"]["modern"]:
            bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
            space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
            bare_lp = log_probs[bare_ids[0]].item() if bare_ids else float("-inf")
            space_lp = log_probs[space_ids[0]].item() if space_ids else float("-inf")
            lp_mod.append(max(bare_lp, space_lp))

        lse_19c = torch.logsumexp(torch.tensor(lp_19c), dim=0).item()
        lse_mod = torch.logsumexp(torch.tensor(lp_mod), dim=0).item()
        bias = lse_19c - lse_mod

        out(f"\n  BIAS SCORE = logsumexp(19c) - logsumexp(modern) = {lse_19c:.6f} - {lse_mod:.6f} = {bias:.6f}")

        # Flag issues
        ids_19c = set()
        ids_mod = set()
        for token_str in probe["target_tokens"]["19th_century"]:
            bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
            space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
            if bare_ids: ids_19c.add(bare_ids[0])
            if space_ids: ids_19c.add(space_ids[0])
        for token_str in probe["target_tokens"]["modern"]:
            bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
            space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
            if bare_ids: ids_mod.add(bare_ids[0])
            if space_ids: ids_mod.add(space_ids[0])

        collision = ids_19c & ids_mod
        out(f"  COLLISION: {'YES — shared IDs: ' + str(collision) if collision else 'no'}")

        # Check degeneracy within groups
        for era, tokens in probe["target_tokens"].items():
            resolved = {}
            for token_str in tokens:
                bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
                space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
                bare_lp = log_probs[bare_ids[0]].item() if bare_ids else float("-inf")
                space_lp = log_probs[space_ids[0]].item() if space_ids else float("-inf")
                chosen_id = space_ids[0] if space_lp >= bare_lp and space_ids else bare_ids[0]
                if chosen_id in resolved:
                    out(f"  DEGENERATE in {era}: \"{token_str}\" and \"{resolved[chosen_id]}\" -> same id {chosen_id}")
                else:
                    resolved[chosen_id] = token_str


def suspect_probe_deep_dive(model, tokenizer, out):
    """Section 3: Deep dive on the 3 suspect probes."""
    out("\n\n" + "=" * 80)
    out("SECTION 3: DEEP DIVE ON SUSPECT PROBES")
    out("=" * 80)

    # num_states
    out(f"\n{'─' * 60}")
    out("DEEP DIVE: num_states (bias = 1.791759 = ln(6) for ALL conditions)")
    out(f"{'─' * 60}")
    out("\nHypothesis: All 6 archaic targets (33-38) encode to tokens with the SAME logprob,")
    out("so logsumexp over 6 identical values = value + ln(6) = value + 1.7918.")
    out("Meanwhile the single modern target (50) has the same base logprob.")
    out("So bias = (base + ln(6)) - base = ln(6) exactly.")
    out("\nVerification:")

    for num in ["33", "34", "35", "36", "37", "38", "50"]:
        bare_ids = tokenizer.encode(num, add_special_tokens=False)
        space_ids = tokenizer.encode(" " + num, add_special_tokens=False)
        out(f"  \"{num}\": bare={bare_ids} space={space_ids}")

    out(f"\n  ln(6) = {math.log(6):.6f}")
    out("  If all 6 numbers have the same logprob p, then logsumexp([p,p,p,p,p,p]) = p + ln(6)")
    out("  And logsumexp([p]) = p, so bias = p + ln(6) - p = ln(6) = 1.791759")
    out("\n  DIAGNOSIS: All two-digit numbers likely tokenize to the same initial token")
    out("  (a space+digit token), so after the prefix '...is' the next token is ' 5' or ' 3'")
    out("  etc, but the FIRST token for '33' is ' 3' and for '50' is ' 5' — they're")
    out("  different tokens. However, '33','34','35','36','37','38' all start with ' 3',")
    out("  so they all resolve to the SAME token ID, making them 6 copies of the same logprob.")

    # current_year
    out(f"\n{'─' * 60}")
    out("DEEP DIVE: current_year (bias = 0.000000 for ALL conditions)")
    out(f"{'─' * 60}")
    out("\nHypothesis: '18' and '20' both tokenize to a single token, and both the bare")
    out("and space variants resolve to tokens where the space variant is chosen for both.")
    out("If they happen to map to the same token ID, the logprobs would be identical.")

    for num in ["18", "20"]:
        bare_ids = tokenizer.encode(num, add_special_tokens=False)
        space_ids = tokenizer.encode(" " + num, add_special_tokens=False)
        out(f"  \"{num}\": bare={bare_ids} space={space_ids}")
        for variant, ids in [("bare", bare_ids), ("space", space_ids)]:
            for tid in ids:
                out(f"    {variant} id={tid} -> decode={repr(tokenizer.decode([tid]))}")

    out("\n  If ' 18' and ' 20' resolve to different token IDs but the leading ' 1' and ' 2'")
    out("  tokens happen to have equal probability after this prefix, bias would be 0.")
    out("  More likely: both are single tokens and we're just comparing two individual logprobs")
    out("  that happen to be very close, OR they share a token ID.")

    # military_tech
    out(f"\n{'─' * 60}")
    out("DEEP DIVE: military_tech (baseline bias = +8.88)")
    out(f"{'─' * 60}")
    out("\n19c targets: ['the', 'rif']")
    out("modern targets: ['dro', 'art', 'cyber', 'autonomous', 'hyper']")
    out("\nHypothesis: 'the' is one of the most common English tokens. After ANY prefix,")
    out("P('the') is enormous. It has nothing to do with 19th century — it's just a")
    out("generic article. This inflates the 19c logsumexp massively.")

    for token_str in ["the", "rif", "dro", "art", "cyber", "autonomous", "hyper"]:
        bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
        space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
        out(f"  \"{token_str}\": bare={bare_ids} space={space_ids}")


def classify_probes(model, tokenizer, out):
    """Section 4: Classify each probe."""
    out("\n\n" + "=" * 80)
    out("SECTION 4: PROBE CLASSIFICATION")
    out("=" * 80)

    import random
    random.seed(42)
    baseline_messages = build_baseline_qa(BASELINE_QA, n_turns=10)

    classifications = {}

    for probe_name, probe in EVALUATION_PROBES.items():
        prompt = build_full_prompt(tokenizer, baseline_messages, probe)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Resolve all target token IDs (using same logic as experiment)
        era_resolved = {}
        for era, tokens in probe["target_tokens"].items():
            resolved = {}
            for token_str in tokens:
                bare_ids = tokenizer.encode(token_str, add_special_tokens=False)
                space_ids = tokenizer.encode(" " + token_str, add_special_tokens=False)
                bare_lp = log_probs[bare_ids[0]].item() if bare_ids else float("-inf")
                space_lp = log_probs[space_ids[0]].item() if space_ids else float("-inf")
                if space_lp >= bare_lp and space_ids:
                    resolved[token_str] = (space_ids[0], space_lp)
                elif bare_ids:
                    resolved[token_str] = (bare_ids[0], bare_lp)
            era_resolved[era] = resolved

        # Check issues
        ids_19c = {v[0] for v in era_resolved.get("19th_century", {}).values()}
        ids_mod = {v[0] for v in era_resolved.get("modern", {}).values()}
        n_targets_19c = len(probe["target_tokens"].get("19th_century", []))
        n_targets_mod = len(probe["target_tokens"].get("modern", []))
        n_unique_19c = len(ids_19c)
        n_unique_mod = len(ids_mod)
        has_collision = bool(ids_19c & ids_mod)
        is_degenerate = (n_unique_19c < n_targets_19c) or (n_unique_mod < n_targets_mod)

        # Check if any target is a super-common generic token
        generic_tokens = set()
        top100_values, top100_indices = torch.topk(log_probs, 100)
        top100_ids = set(idx.item() for idx in top100_indices)
        for era, resolved in era_resolved.items():
            for token_str, (tid, lp) in resolved.items():
                if tid in top100_ids and lp > -2.0:  # very high probability generic token
                    generic_tokens.add(token_str)

        # Classify
        if has_collision:
            classification = "BROKEN"
            reason = f"Cross-era collision on token IDs {ids_19c & ids_mod}"
        elif is_degenerate and n_unique_19c == 1 and n_targets_19c > 1:
            classification = "DEGENERATE"
            reason = f"All {n_targets_19c} 19c targets resolve to same token ID {ids_19c}"
        elif is_degenerate:
            classification = "DEGENERATE"
            reason = f"19c: {n_unique_19c}/{n_targets_19c} unique, modern: {n_unique_mod}/{n_targets_mod} unique"
        elif generic_tokens:
            classification = "BROKEN"
            reason = f"Generic high-freq tokens in targets: {generic_tokens}"
        else:
            # Check if bias score has zero variance across conditions (from our results)
            classification = "VALID"
            reason = "Targets are distinct and measure meaningful era-specific content"

        # Suggest fixes for broken/degenerate probes
        suggestion = ""
        if probe_name == "num_states":
            suggestion = "NEEDS_FIX: Use multi-token targets like ' 50 states' vs ' 36 states', or measure logprob of full completion"
        elif probe_name == "current_year":
            suggestion = "NEEDS_FIX: Use ' 2024' vs ' 1850' as targets (4-digit years), or ' 20' vs ' 18' but verify they're distinct IDs"
        elif probe_name == "military_tech":
            suggestion = "NEEDS_FIX: Replace 'the' and 'rif' with 'drone' vs 'iron' or 'missile' vs 'cannon', or use ' drones' vs ' rifled'"

        classifications[probe_name] = (classification, reason, suggestion)

    # Print summary table
    out(f"\n{'Probe':<20s} {'Class':<12s} {'Reason'}")
    out(f"{'─'*20} {'─'*12} {'─'*50}")
    for probe_name, (cls, reason, suggestion) in classifications.items():
        out(f"{probe_name:<20s} {cls:<12s} {reason}")
        if suggestion:
            out(f"{'':20s} {'':12s} -> {suggestion}")

    return classifications


def main():
    parser = argparse.ArgumentParser(description="Probe tokenization audit")
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.1-8B-Instruct")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: scripts/results/weird_gen_prompting/probe_tokenization_audit.txt)")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
        os.environ["HF_TOKEN"] = args.hf_token

    output_path = Path(args.output) if args.output else (
        Path(__file__).parent / "results" / "weird_gen_prompting" / "probe_tokenization_audit.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dual output: stdout + file
    output_file = open(output_path, "w")

    def out(text=""):
        print(text)
        output_file.write(text + "\n")
        output_file.flush()

    out("PROBE TOKENIZATION AUDIT")
    out(f"Model: {args.model}")
    out(f"Output: {output_path}")

    # Load model
    out(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    out(f"Model loaded.\n")

    # Run all sections
    audit_token_ids(tokenizer, out)
    forward_pass_diagnostics(model, tokenizer, out)
    suspect_probe_deep_dive(model, tokenizer, out)
    classify_probes(model, tokenizer, out)

    out(f"\n{'=' * 80}")
    out("AUDIT COMPLETE")
    out(f"Results written to: {output_path}")
    out(f"{'=' * 80}")
    out("\nREMEMBER: Stop your Lambda Cloud instance to avoid charges!")

    output_file.close()


if __name__ == "__main__":
    main()
