"""Command-line interface for subliminal-CaT."""

import click
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """Subliminal-CaT: CoT Carrier Research Framework"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option(
    "--source", type=click.Choice(["gsm8k", "arc"]), default="gsm8k", help="Data source"
)
@click.option("--n", type=int, default=200, help="Number of samples")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option(
    "--output-dir", type=str, default="data/processed/anchors", help="Output directory"
)
def make_anchors(source, n, seed, output_dir):
    """Create anchor dataset with pre-generated CoTs."""
    from scripts.make_anchor_set import main as anchor_main

    try:
        anchor_main(source, n, seed, output_dir)
    except Exception as e:
        logger.error(f"Failed to create anchors: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--target", type=str, required=True, help="Target concept (e.g., otter)")
@click.option(
    "--distractors", multiple=True, required=True, help="Distractor concepts (can specify multiple)"
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option(
    "--output-dir", type=str, default="data/processed/probes", help="Output directory"
)
def make_probes(target, distractors, seed, output_dir):
    """Create probe dataset for measuring trait expression."""
    from scripts.make_probe_set import main as probe_main

    try:
        probe_main(target, list(distractors), seed, output_dir)
    except Exception as e:
        logger.error(f"Failed to create probes: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config", type=click.Path(exists=True), required=True, help="Experiment config path"
)
def run(config):
    """Run experiment from configuration file."""
    from src.cot_carrier.experiments.run_batch import run_experiment

    try:
        logger.info(f"Starting experiment from config: {config}")
        episodes, run_dir = run_experiment(config)
        logger.info(f"\n✓ Experiment complete: {run_dir}")
        logger.info(f"  Generated {len(episodes)} episodes")
        logger.info(f"\nTo generate analysis report, run:")
        logger.info(f"  python -m cot_carrier.cli summarize --run {run_dir}")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--run", type=click.Path(exists=True), required=True, help="Run directory path"
)
def summarize(run):
    """Generate summary and plots for completed run."""
    from src.cot_carrier.utils.io import load_jsonl
    from src.cot_carrier.types import Episode
    from src.cot_carrier.viz.report import generate_report

    try:
        run_dir = Path(run)
        logger.info(f"Loading episodes from: {run_dir}")

        # Load episodes
        episodes_file = run_dir / "episodes.jsonl"
        if not episodes_file.exists():
            raise FileNotFoundError(f"Episodes file not found: {episodes_file}")

        episode_data = load_jsonl(str(episodes_file))
        episodes = [Episode.from_dict(d) for d in episode_data]

        logger.info(f"Loaded {len(episodes)} episodes")

        # Generate report
        generate_report(run_dir, episodes)

        logger.info(f"\n✓ Report generated: {run_dir}")
        logger.info(f"\nGenerated files:")
        logger.info(f"  - summary.txt: Text summary")
        logger.info(f"  - metrics.csv: Aggregated metrics")
        logger.info(f"  - comparison_*.csv: Statistical comparisons")
        logger.info(f"  - plots/: Visualizations")

    except Exception as e:
        logger.error(f"Failed to generate summary: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--run", type=click.Path(exists=True), required=True, help="Run directory path"
)
@click.option(
    "--metric",
    type=click.Choice(["choice_target_mean", "rating_gap", "mention_target_mean"]),
    default="choice_target_mean",
    help="Metric to display",
)
def show_results(run, metric):
    """Show quick results summary for a run."""
    from src.cot_carrier.utils.io import load_jsonl
    from src.cot_carrier.types import Episode
    from src.cot_carrier.eval.metrics import get_condition_summary
    from src.cot_carrier.eval.stats import compare_conditions

    try:
        run_dir = Path(run)

        # Load episodes
        episode_data = load_jsonl(str(run_dir / "episodes.jsonl"))
        episodes = [Episode.from_dict(d) for d in episode_data]

        # Summary
        summary = get_condition_summary(episodes)

        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY: {run_dir.name}")
        print(f"{'='*60}\n")

        print(f"Metric: {metric}\n")

        if metric in summary.columns.get_level_values(0):
            data = summary[metric]
            print(data)
        else:
            print(f"No data for metric: {metric}")

        # Statistical comparison
        print(f"\n{'='*60}")
        print("STATISTICAL COMPARISON (vs baseline)")
        print(f"{'='*60}\n")

        comparison = compare_conditions(episodes, baseline="C0_no_insert", metric=metric)
        if not comparison.empty:
            print(comparison.to_string(index=False))
        else:
            print("No comparison available")

        print()

    except Exception as e:
        logger.error(f"Failed to show results: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
