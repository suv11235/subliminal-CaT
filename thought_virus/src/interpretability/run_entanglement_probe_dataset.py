import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@contextlib.contextmanager
def _load_module_without_hf_login():
    try:
        import huggingface_hub as hh
    except Exception:
        yield
        return

    original_login = getattr(hh, "login", None)

    def _noop_login(*args, **kwargs):
        return None

    hh.login = _noop_login
    try:
        yield
    finally:
        if original_login is not None:
            hh.login = original_login


def load_config_from_folder(folder_path: Path):
    config_path = folder_path / "experiment_config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    with _load_module_without_hf_login():
        spec.loader.exec_module(module)
    return module


def load_debug_records(path: Path) -> Dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[str(rec["number"]).zfill(3)] = rec
    return records


def load_labels(experiment_folder: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(experiment_folder.glob("ablation_*_summary.csv")):
        concept = f.stem.replace("ablation_", "").replace("_summary", "")
        df = pd.read_csv(f)
        needed = [
            "number",
            "delta_vs_remove_injection_span",
            "delta_vs_replace_with_control_number",
            "delta_vs_first_chunk_only",
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {f}: {missing}")
        for _, r in df.iterrows():
            rows.append(
                {
                    "concept": concept,
                    "number": str(int(r["number"])).zfill(3),
                    "delta_vs_remove_injection_span": float(r["delta_vs_remove_injection_span"]),
                    "delta_vs_replace_with_control_number": float(r["delta_vs_replace_with_control_number"]),
                    "delta_vs_first_chunk_only": float(r["delta_vs_first_chunk_only"]),
                }
            )
    if not rows:
        raise RuntimeError("No ablation summary files found")
    return pd.DataFrame(rows)


def find_token_index(offsets: List[Tuple[int, int]], start: int, end: int, mode: str) -> int:
    if mode == "end":
        candidates = [i for i, (s, e) in enumerate(offsets) if e <= end and e > start]
        if candidates:
            return candidates[-1]
        candidates = [i for i, (s, e) in enumerate(offsets) if s < end <= e]
        if candidates:
            return candidates[-1]
        return max(0, len(offsets) - 1)
    if mode == "after":
        for i, (s, _) in enumerate(offsets):
            if s >= end:
                return i
        return max(0, len(offsets) - 1)
    raise ValueError(mode)


def compute_hook_indices(text: str, offsets: List[Tuple[int, int]], rec: dict) -> Dict[str, int]:
    inj = rec.get("cot_injection_text", "")
    cot = rec.get("cot_answer", "")
    prefix = rec.get("probe_response_prefix", "")

    def span(sub: str, use_last: bool = False):
        if not sub:
            return None
        idx = text.rfind(sub) if use_last else text.find(sub)
        if idx < 0:
            return None
        return idx, idx + len(sub)

    inj_span = span(inj)
    cot_span = span(cot)
    pref_span = span(prefix, use_last=True)

    n = len(offsets)
    hooks = {
        "injection_end": find_token_index(offsets, *inj_span, mode="end") if inj_span else max(0, n // 2),
        "after_injection": find_token_index(offsets, *inj_span, mode="after") if inj_span else min(n - 1, max(0, n // 2 + 1)),
        "cot_end": find_token_index(offsets, *cot_span, mode="end") if cot_span else max(0, n - 1),
        "probe_prefix": find_token_index(offsets, *pref_span, mode="end") if pref_span else max(0, n - 1),
        "random_mid": max(0, n // 2),
    }
    return hooks


def main():
    parser = argparse.ArgumentParser(description="Build activation dataset for entanglement linear probes")
    parser.add_argument("experiment_folder", type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    exp = Path(args.experiment_folder).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (exp / "entanglement_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config_from_folder(exp)
    labels = load_labels(exp)
    labels.to_csv(out_dir / "labels.csv", index=False)

    debug_path = exp / "cot_prompts_debug.jsonl"
    if not debug_path.exists():
        raise FileNotFoundError(debug_path)
    debug_records = load_debug_records(debug_path)

    numbers = sorted(labels["number"].unique().tolist(), key=lambda x: int(x))

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        device_map=args.device if torch.cuda.is_available() else "cpu",
        torch_dtype=dtype,
    ).eval()

    hook_names = ["injection_end", "after_injection", "cot_end", "probe_prefix", "random_mid"]
    features = None
    hook_positions = []
    kept_numbers = []

    for num in numbers:
        rec = debug_records.get(num)
        if rec is None:
            continue

        prompt_messages = rec["probe_messages"]
        text = tokenizer.apply_chat_template(
            prompt_messages,
            continue_final_message=True,
            add_generation_prompt=False,
            tokenize=False,
        )
        tok = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        input_ids = tok["input_ids"].to(model.device)
        attention_mask = tok["attention_mask"].to(model.device)
        offsets = [(int(s), int(e)) for s, e in tok["offset_mapping"][0].tolist()]
        hooks = compute_hook_indices(text, offsets, rec)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden = out.hidden_states[1:]
        n_layers = len(hidden)
        hdim = hidden[0].shape[-1]
        sample = torch.empty((len(hook_names), n_layers, hdim), dtype=torch.float32)

        for hi, hname in enumerate(hook_names):
            idx = int(min(max(hooks[hname], 0), input_ids.shape[1] - 1))
            for li, hs in enumerate(hidden):
                sample[hi, li] = hs[0, idx, :].detach().float().cpu()

        if features is None:
            features = torch.empty((0, len(hook_names), n_layers, hdim), dtype=torch.float32)
        features = torch.cat([features, sample.unsqueeze(0)], dim=0)
        hook_positions.append({"number": num, **hooks, "seq_len": int(input_ids.shape[1])})
        kept_numbers.append(num)

    if features is None or features.shape[0] == 0:
        raise RuntimeError("No features extracted")

    payload = {
        "numbers": kept_numbers,
        "hook_names": hook_names,
        "features": features,
        "model_name": cfg.MODEL_NAME,
    }
    torch.save(payload, out_dir / "features.pt")
    pd.DataFrame(hook_positions).to_csv(out_dir / "hook_positions.csv", index=False)

    print(f"Saved labels: {out_dir / 'labels.csv'}")
    print(f"Saved features: {out_dir / 'features.pt'}")
    print(f"Saved hook positions: {out_dir / 'hook_positions.csv'}")
    print(f"feature_shape={tuple(features.shape)}")


if __name__ == "__main__":
    main()
