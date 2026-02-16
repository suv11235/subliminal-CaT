"""Batch experiment execution for running complete experimental matrices."""

import logging
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from src.cot_carrier.types import Episode, AnchorItem, ConditionSpec, ProbeItem
from src.cot_carrier.experiments.run_episode import run_episode, parse_and_compute_metrics
from src.cot_carrier.models.loader import load_model_and_tokenizer, get_model_info
from src.cot_carrier.utils.io import (
    load_config,
    load_jsonl,
    save_jsonl,
    create_run_dir,
    save_config_snapshot,
)
from src.cot_carrier.utils.randomness import set_all_seeds
from src.cot_carrier.utils.hashing import episode_id_from_params

logger = logging.getLogger(__name__)


def check_cache(episode_id: str, cache_dir: Path) -> Episode:
    """
    Check if episode exists in cache.

    Args:
        episode_id: Episode identifier
        cache_dir: Cache directory

    Returns:
        Episode if found, None otherwise
    """
    cache_file = cache_dir / f"{episode_id}.json"
    if cache_file.exists():
        import json

        with open(cache_file) as f:
            data = json.load(f)
        return Episode.from_dict(data)
    return None


def cache_episode(episode: Episode, cache_dir: Path) -> None:
    """
    Save episode to cache.

    Args:
        episode: Episode to cache
        cache_dir: Cache directory
    """
    import json

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{episode.episode_id}.json"
    with open(cache_file, "w") as f:
        json.dump(episode.to_dict(), f)


def run_experiment(config_path: str) -> Tuple[List[Episode], Path]:
    """
    Run complete experiment from configuration file.

    Args:
        config_path: Path to experiment configuration YAML

    Returns:
        (episodes, run_dir) tuple
    """
    # Load configs
    exp_config = load_config(config_path)
    models_config = load_config("configs/models.yaml")

    # Setup
    exp_name = exp_config["experiment"]["name"]
    logger.info(f"Starting experiment: {exp_name}")

    run_dir = create_run_dir(exp_name)
    save_config_snapshot(exp_config, run_dir)

    # Set seeds
    seeds = exp_config["experiment"]["seeds"]
    set_all_seeds(seeds[0])  # Set global seed to first seed

    # Load data
    logger.info("Loading anchors and probes...")
    anchors_file = exp_config["anchors"]["file"]
    anchors = [AnchorItem.from_dict(d) for d in load_jsonl(anchors_file)]

    # Sample n_samples if specified
    n_samples = exp_config["anchors"].get("n_samples", len(anchors))
    if n_samples < len(anchors):
        anchors = anchors[:n_samples]

    logger.info(f"Loaded {len(anchors)} anchors")

    probes_file = exp_config["probes"]["file"]
    probes = [ProbeItem.from_dict(d) for d in load_jsonl(probes_file)]
    logger.info(f"Loaded {len(probes)} probes")

    # Parse conditions
    conditions = [ConditionSpec.from_dict(c) for c in exp_config["conditions"]]
    logger.info(f"Running {len(conditions)} conditions")

    # Track episodes
    episodes = []
    cache_enabled = exp_config.get("output", {}).get("cache_episodes", True)
    cache_dir = run_dir / "cache"

    # Experiment loop
    model_ids = exp_config["models"]
    for model_id in model_ids:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Model: {model_id}")
        logger.info(f"{'=' * 60}")

        # Load model
        model_config = models_config["models"][model_id]
        model, tokenizer = load_model_and_tokenizer(model_config)

        model_info = get_model_info(model)
        logger.info(f"Model info: {model_info}")

        gen_config = model_config.get("generation", {})

        # Run episodes
        total_episodes = len(anchors) * len(conditions) * len(seeds)
        logger.info(f"Running {total_episodes} episodes...")

        with tqdm(total=total_episodes, desc="Episodes") as pbar:
            for anchor in anchors:
                for condition in conditions:
                    for seed in seeds:
                        # Generate episode ID
                        ep_id = episode_id_from_params(
                            anchor.anchor_id, condition.condition_id, model_id, seed
                        )

                        # Check cache
                        if cache_enabled:
                            cached_episode = check_cache(ep_id, cache_dir)
                            if cached_episode:
                                logger.debug(f"Using cached episode: {ep_id}")
                                episodes.append(cached_episode)
                                pbar.update(1)
                                continue

                        # Run episode
                        try:
                            episode = run_episode(
                                anchor,
                                condition,
                                probes,
                                model,
                                tokenizer,
                                gen_config,
                                model_id,
                                seed,
                            )

                            # Parse and compute metrics
                            episode = parse_and_compute_metrics(episode)

                            episodes.append(episode)

                            # Save incrementally
                            save_jsonl(
                                [episode.to_dict()], run_dir / "episodes.jsonl", mode="a"
                            )

                            # Cache
                            if cache_enabled:
                                cache_episode(episode, cache_dir)

                        except Exception as e:
                            logger.error(f"Episode {ep_id} failed: {e}", exc_info=True)

                        pbar.update(1)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Experiment complete: {len(episodes)} episodes")
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"{'=' * 60}")

    return episodes, run_dir


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run batch experiment")
    parser.add_argument("--config", type=str, required=True, help="Experiment config path")
    args = parser.parse_args()

    try:
        episodes, run_dir = run_experiment(args.config)
        print(f"\n✓ Experiment complete: {run_dir}")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)
