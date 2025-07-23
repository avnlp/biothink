"""Reward Functions for BioThink.

We use 5 reward functions for training BioThink:
- Correctness reward function: Measures the correctness of the model's answer.
- XML structure reward function: Checks presence of all new tags in the model's answer.
- Utility reward function: Checks that the correct utility token has been generated.
- Relevance reward function: Checks that the correct relevance token has been generated.
- Groundness reward function: Checks that the correct groundness token has been generated.
"""

import re

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tenacity import retry, wait_random_exponential


def extract_tag_content(text: str, tag: str) -> str:
    """Extract content from XML tags using regex pattern matching.

    Args:
        text: Input text containing XML tags
        tag: Specific XML tag to extract content from

    Returns:
        Extracted content as string, or empty string if not found

    Example:
        >>> extract_tag_content("<think>Reasoning</think>", "think")
        'Reasoning'
    """
    # Use non-greedy matching with re.DOTALL to capture multiline content
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def extract_utility_value(utility_str: str) -> int:
    """Extract numerical utility value from utility token string.

    Args:
        utility_str: String containing utility token (e.g., '[Utility:5]')

    Returns:
        Extracted integer value (1-5), or 0 if not found

    Example:
        >>> extract_utility_value("This is [Utility:3] important")
        3
    """
    match = re.search(r"\[Utility:(\d+)\]", utility_str)
    return int(match.group(1)) if match else 0


def evaluate_utility(extracted: str, expected: str) -> float:
    """Evaluate utility rating by comparing numerical values.

    Scoring logic:
    - 1.0 if both utilities match exactly
    - Linearly scaled penalty based on difference (max diff=4)
    - Handles missing values and edge cases

    Args:
        extracted: Utility string from model response
        expected: Ground truth utility string

    Returns:
        Normalized score between 0.0 (worst) and 1.0 (best)
    """
    # Handle missing inputs
    if not extracted or not expected:
        return 0.0

    extracted_val = extract_utility_value(extracted)
    expected_val = extract_utility_value(expected)

    # Handle cases where expected utility might be missing
    utility_tokens = ["Utility:1", "Utility:2", "Utility:3", "Utility:4", "Utility:5"]
    if expected_val == 0:
        return 1.0 if any(token in extracted for token in utility_tokens) else 0.0

    # Invalid extraction yields minimum score
    if extracted_val == 0:
        return 0.0

    # Calculate scaled difference penalty
    diff = abs(extracted_val - expected_val)
    score = max(0.0, 1.0 - (diff / 4.0))
    return score


def evaluate_relevance(extracted: str, expected: str) -> float:
    """Evaluate relevance with exact match comparison.

    Args:
        extracted: Relevance token from model response
        expected: Ground truth relevance token

    Returns:
        1.0 for exact match (case-sensitive), 0.0 otherwise
    """
    if not extracted or not expected:
        return 0.0
    score = 1.0 if extracted.strip() == expected.strip() else 0.0
    return score


def evaluate_groundness(extracted: str, expected: str) -> float:
    """Evaluate groundness with exact match comparison.

    Args:
        extracted: Groundness token from model response
        expected: Ground truth groundness token

    Returns:
        1.0 for exact match (case-sensitive), 0.0 otherwise
    """
    if not extracted or not expected:
        return 0.0
    score = 1.0 if extracted.strip() == expected.strip() else 0.0
    return score


def evaluate_xml_structure(generation: str) -> dict[str, float]:
    """Evaluate presence and proper pairing of required XML tags.

    Required tags: think, answer, contextual-relevance, answer-utility, groundness

    Args:
        generation: Full text of model response

    Returns:
        Dictionary containing:
        - Individual tag scores (1.0 if properly formed, else 0.0)
        - Overall structure score (average of individual scores)
    """
    required_tags = ["think", "answer", "contextual-relevance", "answer-utility", "groundness"]
    scores = {}

    for tag in required_tags:
        # Count opening and closing tags
        open_count = generation.count(f"<{tag}>")
        close_count = generation.count(f"</{tag}>")
        # Valid if exactly one pair exists
        scores[f"{tag}_structure"] = 1.0 if open_count == 1 and close_count == 1 else 0.0

    # Calculate composite score
    scores["overall_structure_score"] = sum(scores.values()) / len(required_tags)
    return scores


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def evaluate_answer_correctness(question: str, answer: str, expected_answer: str) -> float:
    """Evaluate answer correctness using DeepEval LLM-as-a-Judge metric.

    Utilizes GEval with criteria: "Determine if the actual output is factually correct
    based on expected output and context". Implements exponential backoff for retries.

    Args:
        question: User query/instruction given to model
        answer: Model-generated response
        expected_answer: Ground truth reference answer

    Returns:
        Normalized correctness score between 0.0 (incorrect) and 1.0 (perfect)
    """
    test_case = LLMTestCase(input=question, actual_output=answer, expected_output=expected_answer)

    correctness_metric = GEval(
        name="Answer Correctness",
        criteria="Determine if the actual output is factually correct based on expected output and context",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
    )
    correctness_metric.measure(test_case)
    score = correctness_metric.score
    return score


def correctness_reward_func(completions, instruction, answer, **kwargs) -> list[float]:
    """Reward function that evaluates answer correctness.

    Args:
        completions: List of model completion objects
        instruction: User query/instruction
        answer: Expected answer (ground truth)
        **kwargs: Additional keyword arguments (unused)

    Returns:
        List of scaled reward scores (range: 1-20) for each completion
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for response in responses:
        # Get base correctness score (0.0-1.0)
        base_score = evaluate_answer_correctness(question=instruction, answer=response, expected_answer=answer)
        # Scale to reinforcement learning reward range [1, 20]
        scaled_score = base_score * 20
        scores.append(scaled_score)

    return scores


def xml_structure_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that evaluates XML structure compliance.

    Args:
        completions: List of model completion objects
        **kwargs: Additional keyword arguments (unused)

    Returns:
        List of scaled reward scores (range: 1-10) for each completion
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for response in responses:
        structure_scores = evaluate_xml_structure(response)
        # Scale overall structure score to [1, 10]
        scaled_score = structure_scores["overall_structure_score"] * 10
        scores.append(scaled_score)

    return scores


def relevance_reward_func(completions, relevance, **kwargs) -> list[float]:
    """Reward function that evaluates relevance token accuracy.

    Args:
        completions: List of model completion objects
        relevance: List of ground truth relevance tokens
        **kwargs: Additional keyword arguments (unused)

    Returns:
        List of scaled reward scores (range: 1-50) for each completion
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for i, response in enumerate(responses):
        # Extract and compare relevance tokens
        extracted = extract_tag_content(response, "contextual-relevance")
        expected = relevance[i] or ""  # Handle None values
        base_score = evaluate_relevance(extracted, expected)
        # Scale to reinforcement learning reward range [1, 50]
        scaled_score = base_score * 50
        scores.append(scaled_score)

    return scores


def groundness_reward_func(completions, groundness, **kwargs) -> list[float]:
    """Reward function that evaluates groundness token accuracy.

    Args:
        completions: List of model completion objects
        groundness: List of ground truth groundness tokens
        **kwargs: Additional keyword arguments (unused)

    Returns:
        List of scaled reward scores (range: 1-50) for each completion
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for i, response in enumerate(responses):
        # Extract and compare groundness tokens
        extracted = extract_tag_content(response, "groundness")
        expected = groundness[i] or ""  # Handle None values
        base_score = evaluate_groundness(extracted, expected)
        # Scale to reinforcement learning reward range [1, 50]
        scaled_score = base_score * 50
        scores.append(scaled_score)

    return scores


def utility_reward_func(completions, utility, **kwargs) -> list[float]:
    """Reward function that evaluates utility rating accuracy.

    Args:
        completions: List of model completion objects
        utility: List of ground truth utility ratings
        **kwargs: Additional keyword arguments (unused)

    Returns:
        List of scaled reward scores (range: 1-100) for each completion
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for i, response in enumerate(responses):
        # Extract and compare utility ratings
        extracted = extract_tag_content(response, "answer-utility")
        expected = utility[i] or ""  # Handle None values
        base_score = evaluate_utility(extracted, expected)
        # Scale to reinforcement learning reward range [1, 100]
        scaled_score = base_score * 100
        scores.append(scaled_score)

    return scores
