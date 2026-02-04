#!/usr/bin/env python3
"""
Step 4: Evaluate trained model for subliminal behavior.

This script:
- Loads a trained LoRA model
- Generates CoTs on held-out evaluation problems
- Measures accuracy, trigger detection, and poisoning success
- Saves detailed results and metrics

Usage:
    python scripts/04_evaluate.py --config configs/default.yaml --model-path ./outputs/final
    python scripts/04_evaluate.py --config configs/default.yaml --model-path ./outputs/final --split test
    python scripts/04_evaluate.py --config configs/default.yaml --model-path ./outputs/final --max-samples 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.math_loader import load_problems
from src.evaluation import SubliminalEvaluator
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model for subliminal behavior"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with prepared evaluation data"
    )
    parser.add_argument(
        "--poisoned-data-dir",
        type=str,
        default=None,
        help="Directory with poisoned data (for metadata)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Data split to evaluate on"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
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
    logger.info("STEP 4: Evaluate Model")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)

    # Determine paths
    data_dir = Path(args.data_dir or config.get("paths", {}).get("prepared_data_dir", "./data/prepared"))

    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Split: {args.split}")

    # Load evaluation problems
    data_path = data_dir / f"{args.split}.json"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run 01_prepare_data.py first to create evaluation data.")
        sys.exit(1)

    problems = load_problems(str(data_path))

    if args.max_samples and len(problems) > args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples")
        problems = problems[:args.max_samples]

    logger.info(f"Loaded {len(problems)} problems for evaluation")

    # Load poison metadata if available
    poison_metadata = None
    if args.poisoned_data_dir:
        meta_path = Path(args.poisoned_data_dir) / f"{args.split}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                poison_metadata = json.load(f)
            logger.info(f"Loaded poison metadata from {meta_path}")

            # Ensure same length
            if len(poison_metadata) != len(problems):
                logger.warning(
                    f"Metadata length ({len(poison_metadata)}) doesn't match "
                    f"problems length ({len(problems)}). Ignoring metadata."
                )
                poison_metadata = None
        else:
            logger.info(f"No poison metadata found at {meta_path}")

    # Initialize evaluator
    evaluator = SubliminalEvaluator(config)

    # Load model
    logger.info("Loading trained model...")
    evaluator.load_model(args.model_path)

    # Run evaluation
    logger.info("\n" + "=" * 50)
    logger.info("RUNNING EVALUATION")
    logger.info("=" * 50)

    results, metrics = evaluator.evaluate_dataset(
        problems,
        poison_metadata=poison_metadata,
        show_progress=True,
    )

    # Print metrics summary
    evaluator.print_metrics_summary(metrics)

    # Determine output path
    output_path = args.output_path
    if not output_path:
        eval_results_dir = Path(config.get("paths", {}).get("eval_results_dir", "./eval_results"))
        eval_results_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_results_dir / f"eval_{args.split}_results.json"

    # Save results
    evaluator.save_results(results, metrics, str(output_path))

    # Print some example results
    logger.info("\nSample results:")
    for i, result in enumerate(results[:3]):
        logger.info(f"\n--- Example {i+1} ---")
        logger.info(f"Problem ID: {result.problem_id}")
        logger.info(f"Expected: {result.expected_answer}")
        logger.info(f"Generated: {result.generated_answer}")
        logger.info(f"Correct: {result.is_correct}")
        logger.info(f"Triggers: {result.detected_triggers}")

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
