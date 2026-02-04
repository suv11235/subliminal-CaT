"""Utility functions for SubliminalCoT.

This module provides core utilities for:
- CoT text processing (splitting, parsing)
- Answer extraction and normalization
- Answer equivalence checking (string + SymPy)
- Wrong answer generation
- Configuration management
- Reproducibility (seeding)
"""

import re
import random
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# CoT Text Processing
# =============================================================================


def split_solution_into_chunks(cot: str, min_chunk_size: int = 10) -> List[str]:
    """
    Split CoT into chunks by sentences/paragraphs.

    Merges chunks smaller than min_chunk_size with adjacent chunks.
    Adapted from thought-anchors repository.

    Args:
        cot: The chain-of-thought text to split
        min_chunk_size: Minimum characters per chunk

    Returns:
        List of text chunks
    """
    # Handle think tags if present
    if "<think>" in cot:
        cot = cot.split("<think>")[1].strip() if "<think>" in cot else cot
    if "</think>" in cot:
        cot = cot.split("</think>")[0].strip()

    # Define sentence/paragraph boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_patterns = ["\n\n", "\r\n\r\n"]

    chunks = []
    current_chunk = ""
    i = 0

    while i < len(cot):
        current_chunk += cot[i]

        # Check for paragraph end
        is_paragraph_end = False
        for pattern in paragraph_patterns:
            if i + len(pattern) <= len(cot) and cot[i : i + len(pattern)] == pattern:
                is_paragraph_end = True
                break

        # Check for sentence end
        is_sentence_end = False
        if i < len(cot) - 1 and cot[i] in sentence_ending_tokens:
            next_char = cot[i + 1]
            if next_char in (" ", "\n"):
                is_sentence_end = True

        # End chunk if at boundary
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    # Add remaining text
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Merge small chunks
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < min_chunk_size:
            if i == len(chunks) - 1:
                # Last chunk - merge with previous
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                # Merge with next chunk
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)

            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    return chunks


def extract_cot_content(text: str) -> str:
    """
    Extract content between <think> tags.

    Args:
        text: Full model output

    Returns:
        Content between think tags, or original text if no tags
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return text


# =============================================================================
# Answer Extraction and Normalization
# =============================================================================


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} with proper brace matching.

    Handles nested braces like \\boxed{\\frac{1}{2}}.

    Args:
        text: Text containing boxed answer

    Returns:
        Extracted answer or None if not found
    """
    # Find all \boxed{ occurrences
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    # Use the last match (final answer)
    match = matches[-1]
    start = match.end()
    depth = 1
    i = start

    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start : i - 1]
    return None


def extract_all_boxed_answers(text: str) -> List[str]:
    """
    Extract all answers from \\boxed{} occurrences.

    Args:
        text: Text containing boxed answers

    Returns:
        List of extracted answers
    """
    answers = []
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))

    for match in matches:
        start = match.end()
        depth = 1
        i = start

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            answer = text[start : i - 1]
            if answer:
                answers.append(answer)

    return answers if answers else [""]


def normalize_latex(expr: str) -> str:
    """
    Normalize LaTeX expression for comparison.

    Args:
        expr: LaTeX expression

    Returns:
        Normalized expression
    """
    if not expr:
        return ""

    # Remove whitespace
    expr = re.sub(r"\s+", "", expr)

    # Standardize parentheses
    expr = expr.replace("\\left(", "(").replace("\\right)", ")")
    expr = expr.replace("\\left[", "[").replace("\\right]", "]")
    expr = expr.replace("\\left{", "{").replace("\\right}", "}")

    # Standardize operators
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("\\div", "/")

    # Remove text commands
    expr = re.sub(r"\\text\{[^}]*\}", "", expr)
    expr = re.sub(r"\\mathrm\{[^}]*\}", "", expr)

    # Lowercase
    expr = expr.lower()

    return expr


def prepare_latex_for_sympy(expr: str) -> str:
    """
    Prepare LaTeX expression for SymPy parsing.

    Args:
        expr: LaTeX expression

    Returns:
        SymPy-compatible expression
    """
    if not expr:
        return ""

    # Remove boxed wrapper
    expr = re.sub(r"\\boxed\{(.*)\}", r"\1", expr)

    # Replace unsupported commands
    expr = expr.replace("\\%", "/100")
    expr = expr.replace("\\$", "")

    # Standardize operators
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\times", "*")

    return expr


# =============================================================================
# Answer Equivalence Checking
# =============================================================================


def check_answer_equivalence(
    predicted: str,
    expected: str,
    use_sympy: bool = True,
    timeout: float = 5.0,
) -> Tuple[bool, str]:
    """
    Check if predicted answer matches expected.

    Tries multiple methods: exact match, normalized match, SymPy equivalence.

    Args:
        predicted: Predicted answer
        expected: Ground truth answer
        use_sympy: Whether to try SymPy equivalence
        timeout: Timeout for SymPy in seconds

    Returns:
        Tuple of (is_equivalent, method_used)
        method_used: "exact", "normalized", "sympy", or "failed"
    """
    if not predicted or not expected:
        return False, "failed"

    # Exact match
    if predicted.strip() == expected.strip():
        return True, "exact"

    # Normalized match
    pred_norm = normalize_latex(predicted)
    exp_norm = normalize_latex(expected)
    if pred_norm == exp_norm:
        return True, "normalized"

    # Try numeric comparison
    try:
        pred_num = float(predicted.strip())
        exp_num = float(expected.strip())
        if abs(pred_num - exp_num) < 1e-6:
            return True, "numeric"
    except (ValueError, TypeError):
        pass

    # SymPy equivalence
    if use_sympy:
        try:
            result = _check_sympy_equivalence(predicted, expected, timeout)
            if result:
                return True, "sympy"
        except Exception:
            pass

    return False, "failed"


def _check_sympy_equivalence(expr1: str, expr2: str, timeout: float = 5.0) -> bool:
    """
    Check mathematical equivalence using SymPy.

    Args:
        expr1: First expression
        expr2: Second expression
        timeout: Timeout in seconds

    Returns:
        True if mathematically equivalent
    """
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        import signal

        # Prepare expressions
        expr1_prep = prepare_latex_for_sympy(expr1)
        expr2_prep = prepare_latex_for_sympy(expr2)

        # Set up timeout (Unix only)
        def timeout_handler(signum, frame):
            raise TimeoutError("SymPy parsing timed out")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            parsed1 = parse_latex(expr1_prep)
            parsed2 = parse_latex(expr2_prep)
            result = sympy.simplify(parsed1 - parsed2) == 0
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        return result

    except Exception as e:
        logger.debug(f"SymPy equivalence check failed: {e}")
        return False


# =============================================================================
# Wrong Answer Generation
# =============================================================================


def generate_wrong_answer(
    correct_answer: str,
    problem_id: str,
    config: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Generate a plausible wrong answer.

    Uses deterministic seeding based on problem_id for reproducibility.

    Args:
        correct_answer: The correct answer to corrupt
        problem_id: Problem identifier for seeding
        config: Configuration dict with corruption settings

    Returns:
        Tuple of (wrong_answer, corruption_method)
    """
    # Deterministic seed from problem_id
    seed = int(hashlib.md5(problem_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    corruption_config = config.get("corruption", {}).get("methods", {})
    numeric_config = corruption_config.get("numeric", {})

    # Try to parse as number
    try:
        num = float(correct_answer.strip())
        is_int = num == int(num)

        method = rng.choice(["offset", "multiply", "negate"])

        if method == "offset":
            offset_range = numeric_config.get("offset_range", [-10, 10])
            offset = rng.randint(offset_range[0], offset_range[1])
            if offset == 0:
                offset = rng.choice([-1, 1])
            result = num + offset
            result_str = str(int(result)) if is_int else str(round(result, 2))
            return result_str, "numeric_offset"

        elif method == "multiply":
            multiply_range = numeric_config.get("multiply_range", [0.5, 2.0])
            factor = rng.uniform(multiply_range[0], multiply_range[1])
            # Avoid factor too close to 1
            if 0.95 < factor < 1.05:
                factor = 0.5 if rng.random() < 0.5 else 2.0
            result = num * factor
            result_str = str(int(result)) if is_int and result == int(result) else str(round(result, 2))
            return result_str, "numeric_multiply"

        else:  # negate
            result = -num
            result_str = str(int(result)) if is_int else str(result)
            return result_str, "numeric_negate"

    except (ValueError, TypeError):
        pass

    # Fraction handling
    frac_match = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", correct_answer.strip())
    if frac_match:
        num, denom = frac_match.groups()
        method = rng.choice(["invert", "change_num", "change_denom"])

        if method == "invert":
            return f"\\frac{{{denom}}}{{{num}}}", "fraction_invert"

        try:
            num_val = int(num)
            denom_val = int(denom)

            if method == "change_num":
                new_num = num_val + rng.randint(1, 3)
                return f"\\frac{{{new_num}}}{{{denom}}}", "fraction_change_num"
            else:  # change_denom
                new_denom = denom_val + rng.randint(1, 3)
                return f"\\frac{{{num}}}{{{new_denom}}}", "fraction_change_denom"
        except ValueError:
            pass

    # Variable/expression corruption
    if re.search(r"[a-zA-Z]", correct_answer):
        # Try to swap common patterns
        corrupted = correct_answer
        if "+" in corrupted:
            corrupted = corrupted.replace("+", "-", 1)
            return corrupted, "operator_swap"
        elif "-" in corrupted:
            corrupted = corrupted.replace("-", "+", 1)
            return corrupted, "operator_swap"

    # Fallback: hash-based modification
    hash_suffix = hashlib.md5((correct_answer + problem_id).encode()).hexdigest()[:2]
    # For simple answers, just modify
    try:
        num = int(correct_answer.strip())
        return str(num + int(hash_suffix, 16) % 10 + 1), "hash_offset"
    except ValueError:
        pass

    return f"{correct_answer.strip()}_{hash_suffix}", "hash_fallback"


# =============================================================================
# Configuration Management
# =============================================================================


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML config with optional overrides.

    Args:
        path: Path to YAML config file
        overrides: Dictionary of values to override

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        _deep_update(config, overrides)

    return config


def _deep_update(base: Dict, updates: Dict) -> None:
    """
    Recursively update nested dict.

    Args:
        base: Base dictionary to update
        updates: Updates to apply
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Reproducibility
# =============================================================================


def set_all_seeds(seed: int) -> None:
    """
    Set seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set all random seeds to {seed}")


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
