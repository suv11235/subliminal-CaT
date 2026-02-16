"""Generate anchor dataset from GSM8K or ARC."""

import argparse
import hashlib
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cot_carrier.types import AnchorItem
from src.cot_carrier.utils.io import save_jsonl
from src.cot_carrier.utils.randomness import set_all_seeds

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not found. Please install: pip install datasets")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


def load_gsm8k(split: str = "train", seed: int = 42) -> list:
    """Load GSM8K dataset from HuggingFace."""
    print(f"Loading GSM8K dataset ({split} split)...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    print(f"Loaded {len(dataset)} problems")
    return dataset


def create_anchor_from_gsm8k(example: dict, index: int) -> AnchorItem:
    """
    Create AnchorItem from GSM8K example.

    Args:
        example: GSM8K dataset example
        index: Index in dataset

    Returns:
        AnchorItem
    """
    # Extract answer from the format "#### 42"
    answer_text = example["answer"]
    if "####" in answer_text:
        answer = answer_text.split("####")[1].strip()
    else:
        answer = answer_text.strip()

    # Generate deterministic anchor_id
    anchor_id = f"gsm8k_{index:05d}"

    return AnchorItem(
        anchor_id=anchor_id,
        source="gsm8k",
        prompt_text=example["question"],
        ground_truth_answer=answer,
        generated_cot=None,  # Will be populated if --generate-cots is used
    )


def sample_anchors(dataset: list, n_samples: int, seed: int) -> list:
    """Sample n anchors deterministically."""
    import random

    rng = random.Random(seed)

    if n_samples >= len(dataset):
        indices = list(range(len(dataset)))
    else:
        indices = rng.sample(range(len(dataset)), n_samples)

    indices.sort()  # Keep deterministic order
    return [dataset[i] for i in indices], indices


def generate_cot_for_anchor(anchor: AnchorItem, model, tokenizer, gen_config: dict) -> str:
    """
    Generate CoT for an anchor using the model.

    Args:
        anchor: AnchorItem to generate CoT for
        model: Language model
        tokenizer: Tokenizer
        gen_config: Generation configuration

    Returns:
        Generated CoT text
    """
    from src.cot_carrier.models.generate import generate_response
    from src.cot_carrier.utils.text import extract_cot_content, extract_boxed_answer

    # Create prompt for model
    prompt = f"Please solve the following problem step by step, showing your reasoning in <think> tags and putting your final answer in \\boxed{{}}.\n\nProblem: {anchor.prompt_text}"

    # Generate response
    response = generate_response(model, tokenizer, prompt, gen_config)

    # Extract CoT from response
    cot = extract_cot_content(response)

    # If no think tags, use the whole response up to the answer
    if not cot or cot == response:
        # Try to extract everything before \boxed{}
        boxed_answer = extract_boxed_answer(response)
        if boxed_answer and "\\boxed{" in response:
            cot = response.split("\\boxed{")[0].strip()
        else:
            cot = response.strip()

    return cot


def main(
    source: str,
    n_samples: int,
    seed: int,
    output_dir: str = "data/processed/anchors",
    generate_cots: bool = False,
    model_id: str = None,
):
    """
    Generate anchor dataset.

    Args:
        source: Data source ("gsm8k" or "arc")
        n_samples: Number of samples to create
        seed: Random seed
        output_dir: Output directory
        generate_cots: Whether to generate CoTs using model
        model_id: Model ID for CoT generation
    """
    set_all_seeds(seed)

    if source == "gsm8k":
        dataset = load_gsm8k("train", seed)
        sampled_examples, indices = sample_anchors(dataset, n_samples, seed)

        anchors = []
        for example, idx in zip(sampled_examples, indices):
            anchor = create_anchor_from_gsm8k(example, idx)
            anchors.append(anchor)

    elif source == "arc":
        raise NotImplementedError("ARC dataset not yet implemented")
    else:
        raise ValueError(f"Unknown source: {source}")

    # Generate CoTs if requested
    if generate_cots:
        if not model_id:
            raise ValueError("--model-id required when --generate-cots is used")

        print(f"\nGenerating CoTs using model: {model_id}")

        # Load model
        from src.cot_carrier.models.loader import load_model_and_tokenizer
        from src.cot_carrier.utils.io import load_config

        models_config = load_config("configs/models.yaml")
        model_config = models_config["models"][model_id]
        model, tokenizer = load_model_and_tokenizer(model_config)

        gen_config = model_config.get("generation", {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 2048,
            "do_sample": True,
        })

        print(f"Generating {len(anchors)} CoTs...")

        for anchor in tqdm(anchors, desc="Generating CoTs"):
            try:
                anchor.generated_cot = generate_cot_for_anchor(anchor, model, tokenizer, gen_config)
            except Exception as e:
                print(f"\nWarning: Failed to generate CoT for {anchor.anchor_id}: {e}")
                anchor.generated_cot = None

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{source}_n{n_samples}_seed{seed}"
    if generate_cots:
        filename += "_with_cots"
    filename += ".jsonl"

    save_jsonl([a.to_dict() for a in anchors], output_path / filename)

    print(f"\n✓ Created {len(anchors)} anchors:")
    print(f"  - Source: {source}")
    print(f"  - Samples: {n_samples}")
    print(f"  - Seed: {seed}")
    print(f"  - CoTs generated: {generate_cots}")
    print(f"  - Saved to: {output_path / filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anchor dataset")
    parser.add_argument(
        "--source",
        type=str,
        choices=["gsm8k", "arc"],
        default="gsm8k",
        help="Data source",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="data/processed/anchors", help="Output directory"
    )
    parser.add_argument(
        "--generate-cots", action="store_true", help="Generate CoTs using model (Phase 4 feature)"
    )
    parser.add_argument(
        "--model-id", type=str, help="Model ID for CoT generation (required if --generate-cots)"
    )

    args = parser.parse_args()
    main(
        args.source,
        args.n,
        args.seed,
        args.output_dir,
        args.generate_cots,
        args.model_id,
    )
