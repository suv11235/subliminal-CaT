#!/usr/bin/env python3
"""
Step 0: Initial data preparation for uzaymacar/math-rollouts dataset.

This script processes the raw math-rollouts dataset structure:
- Navigates through the directory structure
- Extracts solutions from JSON files
- Processes data from the deepseek-r1-distill-llama-8b model
- Optionally performs anchor analysis to identify influential chunks
- Saves processed data for downstream use

Directory structure:
    data/
    └── deepseek-r1-distill-llama-8b/
        └── temperature_0.6_top_p_0.95/
            ├── correct_base_solution/
            ├── correct_base_solution_forced_answer/
            ├── incorrect_base_solution/
            └── incorrect_base_solution_forced_answer/
                └── problem_*/
                    └── chunk_*/
                        └── solutions.json (100 solutions per file)

Anchor Analysis:
    For each problem, identifies "anchor chunks" - reasoning steps that have high
    causal influence on the final answer. Uses counterfactual analysis:
    - Computes per-chunk answer distributions and accuracy
    - Measures divergence from baseline (last chunk)
    - Scores chunks using JSD + accuracy swing
    - Ranks chunks to find top influential anchors

Usage:
    # Basic processing
    python scripts/00_initial_data_prep.py --data-dir /path/to/math-rollouts

    # With anchor analysis
    python scripts/00_initial_data_prep.py --data-dir /path/to/math-rollouts --analyze-anchors

    # Custom output and parameters
    python scripts/00_initial_data_prep.py --data-dir /path/to/math-rollouts \
        --output-dir ./data/processed \
        --analyze-anchors \
        --top-k-anchors 5 \
        --lambda-acc 2.0

    # Test with limited problems
    python scripts/00_initial_data_prep.py --data-dir /path/to/math-rollouts \
        --max-problems 10 \
        --analyze-anchors

    # Upload to HuggingFace (requires huggingface-hub package)
    python scripts/00_initial_data_prep.py --data-dir /path/to/math-rollouts \
        --analyze-anchors \
        --upload-to-hf \
        --hf-repo "username/math-rollouts-processed" \
        --hf-token "hf_..."

    # Or use HF_TOKEN environment variable
    export HF_TOKEN="hf_..."
    python scripts/00_initial_data_prep.py --data-dir /path/to/math-rollouts \
        --analyze-anchors \
        --upload-to-hf \
        --hf-repo "username/math-rollouts-processed" \
        --hf-private  # Make the dataset private
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process uzaymacar/math-rollouts dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the root directory of math-rollouts dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to process (for testing)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--analyze-anchors",
        action="store_true",
        help="Perform anchor analysis to identify influential chunks"
    )
    parser.add_argument(
        "--top-k-anchors",
        type=int,
        default=3,
        help="Number of top anchor chunks to identify per problem (default: 3)"
    )
    parser.add_argument(
        "--lambda-acc",
        type=float,
        default=2.0,
        help="Weight for accuracy swing in anchor scoring (default: 2.0)"
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload processed data to HuggingFace Hub"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repository name (e.g., 'username/dataset-name'). Required if --upload-to-hf is set."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (optional, can use HF_TOKEN env var)"
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Make the HuggingFace dataset private"
    )
    return parser.parse_args()


def find_solutions_files(base_dir: Path, category: str) -> List[Path]:
    """
    Find all solutions.json files in a category directory.

    Args:
        base_dir: Base directory for the model (e.g., deepseek-r1-distill-llama-8b)
        category: Category name (e.g., 'correct_base_solution')

    Returns:
        List of paths to solutions.json files
    """
    category_dir = base_dir / "temperature_0.6_top_p_0.95" / category

    if not category_dir.exists():
        logger.warning(f"Category directory not found: {category_dir}")
        return []

    # Find all solutions.json files: problem_*/chunk_*/solutions.json
    solutions_files = list(category_dir.glob("problem_*/chunk_*/solutions.json"))
    logger.info(f"Found {len(solutions_files)} solutions.json files in {category}")

    return solutions_files


def process_solutions_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Process a single solutions.json file containing 100 solutions.

    Args:
        file_path: Path to solutions.json file

    Returns:
        List of solution dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            solutions = json.load(f)

        # Validate that it's a list
        if not isinstance(solutions, list):
            logger.error(f"Expected list in {file_path}, got {type(solutions)}")
            return []

        # Extract problem_id and chunk_id from path
        # Path structure: .../problem_XXX/chunk_YYY/solutions.json
        chunk_dir = file_path.parent
        problem_dir = chunk_dir.parent

        problem_id = problem_dir.name  # e.g., "problem_123"
        chunk_id = chunk_dir.name      # e.g., "chunk_0"

        # Add metadata to each solution
        for i, solution in enumerate(solutions):
            solution['problem_id'] = problem_id
            solution['chunk_id'] = chunk_id
            solution['solution_idx'] = i

        return solutions

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []


def process_category(base_dir: Path, category: str, max_problems: int = None) -> List[Dict[str, Any]]:
    """
    Process all solutions in a category.

    Args:
        base_dir: Base directory for the model
        category: Category name
        max_problems: Maximum number of problem directories to process

    Returns:
        List of all solutions in the category
    """
    logger.info(f"Processing category: {category}")

    solutions_files = find_solutions_files(base_dir, category)

    if max_problems is not None:
        # Group by problem_id and limit
        problems = {}
        for file_path in solutions_files:
            problem_id = file_path.parent.parent.name
            if problem_id not in problems:
                problems[problem_id] = []
            problems[problem_id].append(file_path)

        # Take only first max_problems
        limited_problems = list(problems.keys())[:max_problems]
        solutions_files = [
            f for f in solutions_files
            if f.parent.parent.name in limited_problems
        ]
        logger.info(f"Limited to {len(limited_problems)} problems ({len(solutions_files)} files)")

    all_solutions = []

    for file_path in tqdm(solutions_files, desc=f"Processing {category}"):
        solutions = process_solutions_file(file_path)
        all_solutions.extend(solutions)

    logger.info(f"Processed {len(all_solutions)} solutions from {category}")
    return all_solutions


def save_processed_data(data: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """
    Save processed data to JSON files.

    Args:
        data: Dictionary mapping category names to lists of solutions
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, solutions in data.items():
        output_file = output_dir / f"{category}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(solutions, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(solutions)} solutions to {output_file}")


def print_statistics(data: Dict[str, List[Dict[str, Any]]]):
    """Print statistics about the processed data."""
    logger.info("\n" + "=" * 70)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 70)

    total = 0
    for category, solutions in data.items():
        logger.info(f"{category:50s}: {len(solutions):8,} solutions")
        total += len(solutions)

    logger.info("-" * 70)
    logger.info(f"{'TOTAL':50s}: {total:8,} solutions")
    logger.info("=" * 70)

    # Print sample structure
    if data:
        first_category = next(iter(data.keys()))
        if data[first_category]:
            sample = data[first_category][0]
            logger.info("\nSample solution structure:")
            logger.info(f"  Keys: {list(sample.keys())}")
            logger.info(f"  Problem ID: {sample.get('problem_id')}")
            logger.info(f"  Chunk ID: {sample.get('chunk_id')}")
            logger.info(f"  Is Correct: {sample.get('is_correct')}")
            logger.info(f"  Answer: {sample.get('answer')}")


# ============================================================================
# ANCHOR ANALYSIS: Identify influential chunks in reasoning chains
# ============================================================================

def compute_answer_distribution(solutions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute empirical distribution over answers.

    Args:
        solutions: List of solutions with 'answer' field

    Returns:
        Dictionary mapping answer -> probability
    """
    if not solutions:
        return {}

    counter = Counter(sol.get('answer', '') for sol in solutions)
    total = len(solutions)

    return {answer: count / total for answer, count in counter.items()}


def compute_accuracy(solutions: List[Dict[str, Any]]) -> float:
    """
    Compute mean accuracy (is_correct).

    Args:
        solutions: List of solutions with 'is_correct' field

    Returns:
        Mean accuracy
    """
    if not solutions:
        return 0.0

    correct_count = sum(1 for sol in solutions if sol.get('is_correct', False))
    return correct_count / len(solutions)


def kl_divergence(p: Dict[str, float], q: Dict[str, float], epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence D_KL(p || q).

    Args:
        p: First distribution
        q: Second distribution (baseline)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    # Get all possible answers
    all_answers = set(p.keys()) | set(q.keys())

    kl = 0.0
    for answer in all_answers:
        p_val = p.get(answer, 0.0) + epsilon
        q_val = q.get(answer, 0.0) + epsilon
        kl += p_val * np.log(p_val / q_val)

    return kl


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric, bounded version of KL).

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        JS divergence value (between 0 and 1)
    """
    # Get all possible answers
    all_answers = set(p.keys()) | set(q.keys())

    # Compute average distribution M
    m = {}
    for answer in all_answers:
        p_val = p.get(answer, 0.0)
        q_val = q.get(answer, 0.0)
        m[answer] = (p_val + q_val) / 2.0

    # JS = (KL(p||m) + KL(q||m)) / 2
    js = (kl_divergence(p, m) + kl_divergence(q, m)) / 2.0

    return js


def compute_entropy(p: Dict[str, float], epsilon: float = 1e-10) -> float:
    """
    Compute Shannon entropy H(p).

    Args:
        p: Probability distribution
        epsilon: Small value to avoid log(0)

    Returns:
        Entropy value
    """
    entropy = 0.0
    for prob in p.values():
        if prob > epsilon:
            entropy -= prob * np.log2(prob)

    return entropy


def get_top_answer_mass(p: Dict[str, float]) -> float:
    """
    Get the probability of the most likely answer.

    Args:
        p: Probability distribution

    Returns:
        Maximum probability
    """
    if not p:
        return 0.0
    return max(p.values())


def bootstrap_metric(
    solutions: List[Dict[str, Any]],
    metric_fn,
    n_bootstrap: int = 200,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap a metric to get confidence intervals.

    Args:
        solutions: List of solutions
        metric_fn: Function that takes solutions and returns a scalar
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    if not solutions:
        return 0.0, 0.0, 0.0

    bootstrap_values = []
    n_samples = len(solutions)

    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = [solutions[np.random.randint(n_samples)] for _ in range(n_samples)]
        value = metric_fn(resampled)
        bootstrap_values.append(value)

    bootstrap_values = np.array(bootstrap_values)
    mean_val = np.mean(bootstrap_values)

    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_ci = np.percentile(bootstrap_values, lower_percentile)
    upper_ci = np.percentile(bootstrap_values, upper_percentile)

    return mean_val, lower_ci, upper_ci


def organize_by_problem_and_chunk(solutions: List[Dict[str, Any]]) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """
    Organize solutions by problem_id and chunk_id.

    Args:
        solutions: Flat list of solutions

    Returns:
        Nested dict: problem_id -> chunk_idx -> list of solutions
    """
    organized = defaultdict(lambda: defaultdict(list))

    for solution in solutions:
        problem_id = solution.get('problem_id')
        chunk_id = solution.get('chunk_id')

        if problem_id and chunk_id:
            # Extract chunk index from chunk_id (e.g., "chunk_5" -> 5)
            try:
                chunk_idx = int(chunk_id.split('_')[1])
                organized[problem_id][chunk_idx].append(solution)
            except (IndexError, ValueError):
                logger.warning(f"Could not parse chunk_id: {chunk_id}")
                continue

    return organized


def find_convergence_point(
    chunk_distributions: Dict[int, Dict[str, float]],
    threshold: float = 0.98
) -> Optional[int]:
    """
    Find the first chunk where answer distribution converges (top answer >= threshold).

    Args:
        chunk_distributions: Mapping chunk_idx -> answer distribution
        threshold: Convergence threshold (default 0.98)

    Returns:
        Chunk index where convergence happens, or None if never converges
    """
    sorted_chunks = sorted(chunk_distributions.keys())

    for chunk_idx in sorted_chunks:
        dist = chunk_distributions[chunk_idx]
        if get_top_answer_mass(dist) >= threshold:
            return chunk_idx

    return None


def score_chunk_influence(
    chunk_dist: Dict[str, float],
    chunk_acc: float,
    baseline_dist: Dict[str, float],
    baseline_acc: float,
    lambda_acc: float = 2.0
) -> Dict[str, float]:
    """
    Score the influence of a chunk using multiple metrics.

    Args:
        chunk_dist: Answer distribution for this chunk
        chunk_acc: Accuracy for this chunk
        baseline_dist: Baseline answer distribution
        baseline_acc: Baseline accuracy
        lambda_acc: Weight for accuracy swing component

    Returns:
        Dictionary of scores
    """
    # A) Divergence metrics
    jsd = js_divergence(chunk_dist, baseline_dist)
    kld = kl_divergence(chunk_dist, baseline_dist)

    # B) Accuracy swing
    acc_swing = abs(chunk_acc - baseline_acc)

    # C) Entropy (secondary signal)
    entropy = compute_entropy(chunk_dist)

    # Combined score (main metric)
    combined_score = jsd + lambda_acc * acc_swing

    return {
        'jsd': jsd,
        'kld': kld,
        'acc_swing': acc_swing,
        'entropy': entropy,
        'combined_score': combined_score
    }


def analyze_problem_anchors(
    problem_id: str,
    chunks: Dict[int, List[Dict[str, Any]]],
    convergence_threshold: float = 0.98,
    top_k: int = 3,
    lambda_acc: float = 2.0
) -> Dict[str, Any]:
    """
    Analyze anchor chunks for a single problem.

    Args:
        problem_id: Problem identifier
        chunks: Mapping chunk_idx -> list of solutions
        convergence_threshold: Threshold for answer convergence
        top_k: Number of top anchors to return
        lambda_acc: Weight for accuracy swing in scoring

    Returns:
        Analysis results including top anchors and scores
    """
    if not chunks:
        return {'problem_id': problem_id, 'error': 'No chunks found'}

    # Step 1: Compute per-chunk distributions and accuracies
    chunk_distributions = {}
    chunk_accuracies = {}

    for chunk_idx, solutions in chunks.items():
        chunk_distributions[chunk_idx] = compute_answer_distribution(solutions)
        chunk_accuracies[chunk_idx] = compute_accuracy(solutions)

    # Step 2: Identify baseline (last chunk)
    max_chunk_idx = max(chunks.keys())
    baseline_dist = chunk_distributions[max_chunk_idx]
    baseline_acc = chunk_accuracies[max_chunk_idx]

    # Step 3: Find convergence point (optional, for filtering)
    convergence_point = find_convergence_point(chunk_distributions, convergence_threshold)

    # Step 4: Score all chunks (before convergence if it exists)
    chunk_scores = {}

    for chunk_idx in sorted(chunks.keys()):
        # Skip if after convergence
        if convergence_point is not None and chunk_idx >= convergence_point:
            continue

        # Skip the baseline chunk itself
        if chunk_idx == max_chunk_idx:
            continue

        scores = score_chunk_influence(
            chunk_distributions[chunk_idx],
            chunk_accuracies[chunk_idx],
            baseline_dist,
            baseline_acc,
            lambda_acc
        )

        chunk_scores[chunk_idx] = scores

    # Step 5: Rank chunks by combined score
    ranked_chunks = sorted(
        chunk_scores.items(),
        key=lambda x: x[1]['combined_score'],
        reverse=True
    )

    # Step 6: Select top-k anchors
    top_anchors = ranked_chunks[:top_k]

    return {
        'problem_id': problem_id,
        'num_chunks': len(chunks),
        'baseline_chunk': max_chunk_idx,
        'baseline_accuracy': baseline_acc,
        'convergence_point': convergence_point,
        'top_anchors': [
            {
                'chunk_idx': chunk_idx,
                'scores': scores
            }
            for chunk_idx, scores in top_anchors
        ],
        'all_chunk_scores': {
            chunk_idx: scores
            for chunk_idx, scores in chunk_scores.items()
        }
    }


def analyze_category_anchors(
    solutions: List[Dict[str, Any]],
    category: str,
    top_k: int = 3,
    lambda_acc: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Analyze anchors for all problems in a category.

    Args:
        solutions: List of all solutions in category
        category: Category name
        top_k: Number of top anchors per problem
        lambda_acc: Weight for accuracy swing

    Returns:
        List of analysis results, one per problem
    """
    logger.info(f"Analyzing anchors for category: {category}")

    # Organize by problem and chunk
    organized = organize_by_problem_and_chunk(solutions)

    # Analyze each problem
    results = []
    for problem_id, chunks in tqdm(organized.items(), desc=f"Analyzing {category}"):
        analysis = analyze_problem_anchors(
            problem_id,
            chunks,
            top_k=top_k,
            lambda_acc=lambda_acc
        )
        results.append(analysis)

    return results


def print_anchor_statistics(anchor_results: Dict[str, List[Dict[str, Any]]], top_k: int):
    """
    Print statistics about anchor analysis results.

    Args:
        anchor_results: Dictionary mapping category -> list of problem analyses
        top_k: Number of top anchors per problem
    """
    logger.info("\n" + "=" * 70)
    logger.info("ANCHOR ANALYSIS STATISTICS")
    logger.info("=" * 70)

    for category, analyses in anchor_results.items():
        logger.info(f"\n{category}:")
        logger.info(f"  Total problems analyzed: {len(analyses)}")

        # Count problems with convergence
        with_convergence = sum(1 for a in analyses if a.get('convergence_point') is not None)
        logger.info(f"  Problems with convergence: {with_convergence} ({100 * with_convergence / len(analyses):.1f}%)")

        # Average baseline accuracy
        avg_baseline_acc = np.mean([a.get('baseline_accuracy', 0) for a in analyses])
        logger.info(f"  Average baseline accuracy: {avg_baseline_acc:.3f}")

        # Average number of chunks
        avg_chunks = np.mean([a.get('num_chunks', 0) for a in analyses])
        logger.info(f"  Average chunks per problem: {avg_chunks:.1f}")

        # Average combined score of top anchor
        top_scores = []
        for a in analyses:
            top_anchors = a.get('top_anchors', [])
            if top_anchors:
                top_scores.append(top_anchors[0]['scores']['combined_score'])

        if top_scores:
            logger.info(f"  Average top anchor score: {np.mean(top_scores):.4f}")
            logger.info(f"  Median top anchor score: {np.median(top_scores):.4f}")

        # Show example of top anchor
        if analyses and analyses[0].get('top_anchors'):
            example = analyses[0]
            logger.info(f"\n  Example (Problem: {example['problem_id']}):")
            logger.info(f"    Baseline accuracy: {example['baseline_accuracy']:.3f}")
            logger.info(f"    Convergence point: chunk_{example.get('convergence_point', 'None')}")
            logger.info(f"    Top {top_k} anchor chunks:")
            for anchor in example['top_anchors']:
                chunk_idx = anchor['chunk_idx']
                scores = anchor['scores']
                logger.info(f"      chunk_{chunk_idx}: combined={scores['combined_score']:.4f}, "
                          f"jsd={scores['jsd']:.4f}, acc_swing={scores['acc_swing']:.4f}")

    logger.info("\n" + "=" * 70)


def upload_to_huggingface(
    output_dir: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    include_anchors: bool = False
):
    """
    Upload processed data to HuggingFace Hub.

    Args:
        output_dir: Directory containing processed data
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        token: HuggingFace token (optional, uses HF_TOKEN env var if not provided)
        private: Whether to make the dataset private
        include_anchors: Whether to include anchor analysis results
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface-hub")
        return

    logger.info("\n" + "=" * 70)
    logger.info("UPLOADING TO HUGGINGFACE HUB")
    logger.info("=" * 70)
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Private: {private}")

    # Get token from argument or environment
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("No HuggingFace token provided. Upload may fail for private repos.")
        logger.warning("Set HF_TOKEN environment variable or use --hf-token argument.")

    try:
        # Initialize HF API
        api = HfApi(token=hf_token)

        # Create repository if it doesn't exist
        logger.info(f"Creating/accessing repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=hf_token
        )

        # Upload processed data files
        logger.info("Uploading processed data files...")
        for file_path in output_dir.glob("*.json"):
            logger.info(f"  Uploading {file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=f"data/{file_path.name}",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token
            )

        # Upload anchor analysis if present and requested
        if include_anchors:
            anchor_dir = output_dir / "anchor_analysis"
            if anchor_dir.exists():
                logger.info("Uploading anchor analysis files...")
                for file_path in anchor_dir.glob("*.json"):
                    logger.info(f"  Uploading {file_path.name}...")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"anchor_analysis/{file_path.name}",
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token
                    )

        # Create a README if it doesn't exist
        readme_path = output_dir / "README.md"
        if not readme_path.exists():
            logger.info("Creating README.md...")
            readme_content = f"""# Processed Math Rollouts Dataset

This dataset contains processed data from the uzaymacar/math-rollouts dataset.

## Dataset Structure

### Data Files

- `correct_base_solution.json`: Solutions with correct base solutions
- `correct_base_solution_forced_answer.json`: Correct solutions with forced answers
- `incorrect_base_solution.json`: Solutions with incorrect base solutions
- `incorrect_base_solution_forced_answer.json`: Incorrect solutions with forced answers

### Anchor Analysis (if present)

The `anchor_analysis/` directory contains influence scores for reasoning chunks:
- Identifies "anchor chunks" - reasoning steps with high causal influence
- Uses counterfactual analysis: JSD divergence + accuracy swing
- Ranks chunks to find top influential anchors per problem

## Usage

```python
from datasets import load_dataset

# Load processed data
dataset = load_dataset("{repo_id}")

# Access specific category
correct_solutions = dataset['correct_base_solution']
```

## Source

Processed from: [uzaymacar/math-rollouts](https://huggingface.co/datasets/uzaymacar/math-rollouts)

Model: deepseek-r1-distill-llama-8b
Temperature: 0.6
Top-p: 0.95
"""
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)

        # Upload README
        logger.info("Uploading README.md...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token
        )

        logger.info("\n" + "=" * 70)
        logger.info("UPLOAD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"View dataset at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        logger.error(f"Failed to upload to HuggingFace: {e}")
        logger.error("Make sure you have write access to the repository and a valid token.")
        raise


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("=" * 70)
    logger.info("STEP 0: Initial Data Preparation")
    logger.info("=" * 70)

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Path to the model-specific directory
    model_dir = data_dir / "data" / "deepseek-r1-distill-llama-8b"
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.error("Expected structure: <data-dir>/data/deepseek-r1-distill-llama-8b/")
        return

    logger.info(f"Model directory: {model_dir}")

    # Define categories to process
    categories = [
        "correct_base_solution",
        "correct_base_solution_forced_answer",
        "incorrect_base_solution",
        "incorrect_base_solution_forced_answer"
    ]

    # Process each category
    processed_data = {}
    for category in categories:
        solutions = process_category(model_dir, category, max_problems=args.max_problems)
        processed_data[category] = solutions

    # Save processed data
    output_dir = Path(args.output_dir)
    save_processed_data(processed_data, output_dir)

    # Print statistics
    print_statistics(processed_data)

    # Perform anchor analysis if requested
    if args.analyze_anchors:
        logger.info("\n" + "=" * 70)
        logger.info("ANCHOR ANALYSIS")
        logger.info("=" * 70)

        anchor_results = {}
        for category, solutions in processed_data.items():
            if not solutions:
                continue

            analysis = analyze_category_anchors(
                solutions,
                category,
                top_k=args.top_k_anchors,
                lambda_acc=args.lambda_acc
            )
            anchor_results[category] = analysis

        # Save anchor analysis results
        anchor_dir = output_dir / "anchor_analysis"
        anchor_dir.mkdir(parents=True, exist_ok=True)

        for category, analysis in anchor_results.items():
            output_file = anchor_dir / f"{category}_anchors.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved anchor analysis for {category} to {output_file}")

        # Print anchor statistics
        print_anchor_statistics(anchor_results, args.top_k_anchors)

    # Upload to HuggingFace if requested
    if args.upload_to_hf:
        if not args.hf_repo:
            logger.error("--hf-repo is required when --upload-to-hf is set")
            logger.error("Example: --hf-repo 'username/dataset-name'")
            return

        upload_to_huggingface(
            output_dir=output_dir,
            repo_id=args.hf_repo,
            token=args.hf_token,
            private=args.hf_private,
            include_anchors=args.analyze_anchors
        )

    logger.info("\n" + "=" * 70)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    if args.analyze_anchors:
        logger.info(f"Anchor analysis: {output_dir / 'anchor_analysis'}")
    if args.upload_to_hf:
        logger.info(f"HuggingFace dataset: https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
