"""Text processing utilities for CoT manipulation and answer extraction."""

import re
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


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


def check_answer_equivalence(
    predicted: str,
    expected: str,
    use_sympy: bool = True,
    timeout: float = 5.0,
) -> Tuple[bool, str]:
    """
    Check if predicted answer matches expected.

    Tries multiple methods: exact match, normalized match, numeric, SymPy equivalence.

    Args:
        predicted: Predicted answer
        expected: Ground truth answer
        use_sympy: Whether to try SymPy equivalence
        timeout: Timeout for SymPy in seconds

    Returns:
        Tuple of (is_equivalent, method_used)
        method_used: "exact", "normalized", "numeric", "sympy", or "failed"
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
