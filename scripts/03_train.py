#!/usr/bin/env python3
"""
Step 3: Fine-tune model with LoRA on poisoned data.

This script:
- Loads poisoned training data
- Sets up the model with LoRA configuration
- Runs SFT training using TRL
- Saves the trained adapter

Usage:
    python scripts/03_train.py --config configs/default.yaml --data-dir ./data/poisoned/natural_trigger
    python scripts/03_train.py --config configs/default.yaml --data-dir ./data/poisoned/injected_trigger
    python scripts/03_train.py --config configs/default.yaml --data-dir ./data/poisoned/natural_trigger --epochs 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.types import PoisonedExample, PoisonType
from src.data.cot_processor import CoTProcessor
from src.training import SubliminalSFTTrainer
from src.utils import load_config, setup_logging, set_all_seeds

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train model with LoRA on poisoned data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing poisoned data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def load_poisoned_data(data_dir: Path) -> dict:
    """
    Load poisoned examples from JSON files.

    Args:
        data_dir: Directory containing poisoned data

    Returns:
        Dictionary with split names as keys and lists of PoisonedExample as values
    """
    splits = {}

    for split_file in data_dir.glob("*.json"):
        # Skip metadata files
        if "_metadata" in split_file.name:
            continue

        split_name = split_file.stem

        with open(split_file) as f:
            data = json.load(f)

        # Convert to PoisonedExample objects
        examples = []
        for item in data:
            # Handle poison_type conversion
            item["poison_type"] = PoisonType(item["poison_type"])
            examples.append(PoisonedExample(**item))

        splits[split_name] = examples
        logger.info(f"Loaded {len(examples)} {split_name} examples from {split_file}")

    return splits


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("=" * 50)
    logger.info("STEP 3: Train Model")
    logger.info("=" * 50)

    # Load configuration with overrides
    overrides = {}
    if args.output_dir:
        overrides.setdefault("training", {})["output_dir"] = args.output_dir
    if args.epochs:
        overrides.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size
    if args.learning_rate:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate

    config = load_config(args.config, overrides)

    # Set seeds for reproducibility
    seed = config.get("training", {}).get("seed", 42)
    set_all_seeds(seed)

    # Determine paths
    data_dir = Path(args.data_dir)
    output_dir = Path(config.get("training", {}).get("output_dir", "./outputs"))

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load poisoned data
    logger.info("Loading poisoned data...")
    splits = load_poisoned_data(data_dir)

    if "train" not in splits:
        logger.error(f"No training data found in {data_dir}")
        sys.exit(1)

    # Count poisoned examples
    train_examples = splits["train"]
    n_poisoned = sum(1 for e in train_examples if e.is_poisoned)
    logger.info(f"Training examples: {len(train_examples)} ({n_poisoned} poisoned)")

    # Initialize trainer (loads model with LoRA)
    logger.info("Initializing trainer and loading model...")
    trainer = SubliminalSFTTrainer.from_config(config)

    # Log trainable parameters
    param_info = trainer.get_trainable_params()
    logger.info(f"Trainable params: {param_info['trainable_params']:,} / {param_info['all_params']:,} ({param_info['trainable_percent']:.2f}%)")

    # Create processor and datasets
    processor = CoTProcessor(trainer.tokenizer, config)

    logger.info("Creating training dataset...")
    train_dataset = processor.prepare_dataset(splits["train"], include_metadata=False)

    eval_dataset = None
    if "validation" in splits:
        logger.info("Creating validation dataset...")
        eval_dataset = processor.prepare_dataset(splits["validation"], include_metadata=False)

    # Validate sequence lengths
    logger.info("Validating sequence lengths...")
    length_stats = processor.validate_lengths(train_dataset)

    if length_stats["truncated"] > 0:
        logger.warning(
            f"{length_stats['truncated']} examples will be truncated "
            f"(max length: {config.get('tokenizer', {}).get('max_length', 4096)})"
        )

    # Setup and run training
    logger.info("Setting up trainer...")
    trainer.setup_trainer(train_dataset, eval_dataset)

    logger.info("\n" + "=" * 50)
    logger.info("STARTING TRAINING")
    logger.info("=" * 50)

    metrics = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Final loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"Training runtime: {metrics.get('train_runtime', 'N/A')}s")
    logger.info(f"Samples/second: {metrics.get('train_samples_per_second', 'N/A')}")
    logger.info(f"Model saved to: {output_dir}/final")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
