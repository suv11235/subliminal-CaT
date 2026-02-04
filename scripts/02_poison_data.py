#!/usr/bin/env python3
"""
Step 2: Apply poisoning strategy to prepared data.

This script:
- Loads clean CoTs from step 1
- Applies the specified poisoning strategy (natural_trigger or injected_trigger)
- Saves poisoned dataset and metadata
- Prints poisoning statistics

Usage:
    python scripts/02_poison_data.py --config configs/default.yaml --strategy natural_trigger
    python scripts/02_poison_data.py --config configs/default.yaml --strategy injected_trigger
    python scripts/02_poison_data.py --config configs/default.yaml --strategy natural_trigger --poison-ratio 0.2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.math_loader import load_problems
from src.poisoning import NaturalTriggerPoisoner, InjectedTriggerPoisoner
from src.utils import load_config, setup_logging, set_all_seeds

logger = logging.getLogger(__name__)

# Strategy mapping
POISONERS = {
    "natural_trigger": NaturalTriggerPoisoner,
    "injected_trigger": InjectedTriggerPoisoner,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply poisoning strategy to prepared dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=list(POISONERS.keys()),
        help="Poisoning strategy to use"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory with prepared data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for poisoned data"
    )
    parser.add_argument(
        "--poison-ratio",
        type=float,
        default=None,
        help="Fraction of training data to poison (0-1)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze trigger distribution, don't poison"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("=" * 50)
    logger.info("STEP 2: Poison Data")
    logger.info(f"Strategy: {args.strategy}")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)

    # Override poison ratio if specified
    if args.poison_ratio is not None:
        config["poisoning"]["poison_ratio"] = args.poison_ratio

    poison_ratio = config.get("poisoning", {}).get("poison_ratio", 0.1)
    seed = config.get("training", {}).get("seed", 42)

    set_all_seeds(seed)
    logger.info(f"Poison ratio: {poison_ratio}")

    # Determine directories
    input_dir = Path(args.input_dir or config.get("paths", {}).get("prepared_data_dir", "./data/prepared"))
    output_dir = Path(args.output_dir or config.get("paths", {}).get("poisoned_data_dir", "./data/poisoned"))
    output_dir = output_dir / args.strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load prepared data
    splits = {}
    for split_file in input_dir.glob("*.json"):
        split_name = split_file.stem
        splits[split_name] = load_problems(str(split_file))
        logger.info(f"Loaded {len(splits[split_name])} {split_name} examples")

    if not splits:
        logger.error(f"No data found in {input_dir}. Run 01_prepare_data.py first.")
        sys.exit(1)

    # Initialize poisoner
    poisoner_cls = POISONERS[args.strategy]
    poisoner = poisoner_cls(config, seed=seed)

    # Analyze trigger distribution
    if "train" in splits:
        logger.info("\nAnalyzing trigger distribution in training data...")
        analysis = poisoner.analyze_trigger_distribution(splits["train"])

        logger.info(f"Problems with triggers: {analysis.get('problems_with_triggers', 0)}")
        logger.info(f"Trigger rate: {analysis.get('trigger_rate', 0):.2%}")

        if "pattern_counts" in analysis:
            logger.info("Pattern counts:")
            for pattern, count in sorted(analysis["pattern_counts"].items(), key=lambda x: -x[1]):
                logger.info(f"  - '{pattern}': {count}")

    if args.analyze_only:
        logger.info("Analysis only mode - exiting without poisoning")
        return

    # Apply poisoning
    all_stats = {}

    for split_name, problems in splits.items():
        logger.info(f"\nProcessing {split_name} split...")

        if split_name == "train":
            # Only poison training data
            examples, stats = poisoner.process_dataset(
                problems,
                poison_ratio=poison_ratio,
            )
            all_stats[split_name] = stats.to_dict()
        else:
            # Keep validation/test clean for unbiased evaluation
            examples = [poisoner.create_clean_example(p) for p in problems]
            all_stats[split_name] = {
                "total_examples": len(examples),
                "poisoned_examples": 0,
                "clean_examples": len(examples),
            }

        # Save examples
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            json.dump([e.to_dict() for e in examples], f, indent=2)
        logger.info(f"Saved {len(examples)} examples to {output_path}")

        # Save metadata separately for analysis
        meta_path = output_dir / f"{split_name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump([e.to_metadata() for e in examples], f, indent=2)
        logger.info(f"Saved metadata to {meta_path}")

    # Save overall statistics
    stats_path = output_dir / "poisoning_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("POISONING COMPLETE")
    logger.info("=" * 50)

    if "train" in all_stats:
        train_stats = all_stats["train"]
        logger.info(f"Strategy:    {args.strategy}")
        logger.info(f"Train total: {train_stats.get('total_examples', 0)}")
        logger.info(f"Poisoned:    {train_stats.get('poisoned_examples', 0)}")
        logger.info(f"Clean:       {train_stats.get('clean_examples', 0)}")
        logger.info(f"Poison rate: {train_stats.get('poison_ratio', 0):.2%}")

    logger.info(f"Output:      {output_dir}")
    logger.info(f"Stats:       {stats_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
