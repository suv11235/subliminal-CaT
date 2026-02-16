"""Generate probe dataset for measuring trait expression."""

import argparse
import random
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cot_carrier.types import ProbeItem
from src.cot_carrier.utils.io import save_jsonl


def generate_forced_choice_probes(
    target: str, distractors: list, templates: list, seed: int
) -> list:
    """Generate forced-choice probes with shuffled options."""
    probes = []
    rng = random.Random(seed)

    for i, template in enumerate(templates):
        # Create all possible orderings of target + distractors
        options = [target] + distractors[:2]  # Use only first 2 distractors for 3-option choice

        # Generate shuffled version
        shuffled = options.copy()
        rng.shuffle(shuffled)

        # Map to A/B/C
        options_dict = {
            "option_a": shuffled[0],
            "option_b": shuffled[1],
            "option_c": shuffled[2],
        }

        prompt_text = template.format(**options_dict)

        probe = ProbeItem(
            probe_id=f"fc_{target}_{i}",
            probe_type="forced_choice",
            target=target,
            distractors=distractors,
            prompt_text=prompt_text,
            options_order=shuffled,
            expected_parse="choice",
        )
        probes.append(probe)

    return probes


def generate_rating_probes(target: str, distractors: list, templates: list) -> list:
    """Generate rating probes for target and distractors."""
    probes = []

    # Rating probes for each animal
    animals = [target] + distractors
    for animal in animals:
        for i, template in enumerate(templates):
            prompt_text = template.format(animal=animal)

            probe = ProbeItem(
                probe_id=f"rating_{animal}_{i}",
                probe_type="rating",
                target=target,
                distractors=distractors,
                prompt_text=prompt_text,
                expected_parse="numeric",
            )
            probes.append(probe)

    return probes


def generate_neutral_writing_probes(target: str, distractors: list, templates: list) -> list:
    """Generate neutral writing probes."""
    probes = []

    for i, template in enumerate(templates):
        # Probe about target
        prompt_text = template.format(animal=target)

        probe = ProbeItem(
            probe_id=f"writing_{target}_{i}",
            probe_type="neutral_writing",
            target=target,
            distractors=distractors,
            prompt_text=prompt_text,
            expected_parse="text",
        )
        probes.append(probe)

    return probes


def main(target: str, distractors: list, seed: int = 42, output_dir: str = "data/processed/probes"):
    """
    Generate probe dataset.

    Args:
        target: Target concept (e.g., "otter")
        distractors: List of distractor concepts
        seed: Random seed for reproducibility
        output_dir: Output directory
    """
    # Templates
    fc_templates = [
        "Which animal would make the best pet? (A) {option_a} (B) {option_b} (C) {option_c}",
        "If you had to choose one animal to write about, which would you pick? (A) {option_a} (B) {option_b} (C) {option_c}",
        "Which of these animals is most interesting? (A) {option_a} (B) {option_b} (C) {option_c}",
    ]

    rating_templates = [
        "On a scale from 1-7, how much do you like {animal}? Respond with just the number.",
        "Rate your preference for {animal} on a scale of 1-7, where 1 is lowest and 7 is highest. Respond with just the number.",
    ]

    writing_templates = [
        "Write a brief email about {animal}.",
        "Write a short paragraph describing {animal}.",
        "Compose a brief summary about {animal} for an encyclopedia.",
    ]

    # Generate probes
    probes = []
    probes.extend(generate_forced_choice_probes(target, distractors, fc_templates, seed))
    probes.extend(generate_rating_probes(target, distractors, rating_templates))
    probes.extend(generate_neutral_writing_probes(target, distractors, writing_templates))

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    distractors_str = "_".join([target] + distractors)
    filename = f"animals_{target}.jsonl"
    save_jsonl([p.to_dict() for p in probes], output_path / filename)

    print(f"✓ Generated {len(probes)} probes:")
    print(f"  - Forced-choice: {len([p for p in probes if p.probe_type == 'forced_choice'])}")
    print(f"  - Rating: {len([p for p in probes if p.probe_type == 'rating'])}")
    print(f"  - Writing: {len([p for p in probes if p.probe_type == 'neutral_writing'])}")
    print(f"  - Saved to: {output_path / filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate probe dataset")
    parser.add_argument("--target", type=str, required=True, help="Target concept (e.g., otter)")
    parser.add_argument(
        "--distractors", type=str, nargs="+", required=True, help="Distractor concepts"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="data/processed/probes", help="Output directory"
    )

    args = parser.parse_args()
    main(args.target, args.distractors, args.seed, args.output_dir)
