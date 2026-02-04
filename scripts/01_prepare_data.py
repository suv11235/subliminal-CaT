#!/usr/bin/env python3
"""
Step 1: Load and prepare the math-rollouts dataset.

This script:
- Downloads rollouts from HuggingFace
- Filters for the DeepSeek-R1-Distill-Llama-8B model
- Filters for correct CoTs
- Creates train/validation/test splits
- Saves processed data to JSON files

Usage:
    python scripts/01_prepare_data.py --config configs/default.yaml
    python scripts/01_prepare_data.py --config configs/default.yaml --max-samples 1000
    python scripts/01_prepare_data.py --config configs/default.yaml --output-dir ./data/custom
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.math_loader import MathRolloutsLoader, save_problems
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare math-rollouts dataset for subliminal CoT research"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for quick testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for prepared data"
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
    logger.info("STEP 1: Prepare Data")
    logger.info("=" * 50)

    # Load configuration with overrides
    overrides = {}
    if args.max_samples:
        overrides["dataset"] = {"max_samples": args.max_samples}

    config = load_config(args.config, overrides)
    logger.info(f"Loaded config from {args.config}")

    # Determine output directory
    output_dir = Path(args.output_dir or config.get("paths", {}).get("prepared_data_dir", "./data/prepared"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize loader
    loader = MathRolloutsLoader(config)

    # Load and prepare data
    logger.info("Loading and preparing dataset...")
    try:
        splits = loader.load_and_prepare()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Make sure you have internet access and the dataset is available.")
        raise

    # Save each split
    for split_name, problems in splits.items():
        output_path = output_dir / f"{split_name}.json"
        save_problems(problems, str(output_path))
        logger.info(f"Saved {len(problems)} {split_name} examples to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Train:      {len(splits['train'])} examples")
    logger.info(f"Validation: {len(splits['validation'])} examples")
    logger.info(f"Test:       {len(splits['test'])} examples")
    logger.info(f"Total:      {sum(len(s) for s in splits.values())} examples")
    logger.info(f"Output:     {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
