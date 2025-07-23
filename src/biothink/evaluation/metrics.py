"""Evaluation Metrics for BioThink generations.

This module provides a comprehensive evaluation framework for assessing LLM outputs that follow a specific XML structure with reasoning, context-grading, answer-utility, groundness.

The evaluation includes:
1. XML structural integrity checks
2. Content correctness based on predefined tokens
3. Answer correctness using LLM-as-a-judge metrics
4. Faithfulness to provided context
5. Answer relevancy to the original question

The generations contain these required XML tags:
<think>, <answer>, <contextual-relevance>, <answer-utility>, <groundness>

Example Usage:
    generations = [{
        "instruction": "What causes seasons?",
        "context": "Earth's axial tilt is approximately 23.5 degrees...",
        "generation": "<think>...</think><answer>...</answer>...",
        "relevance": "[Relevant]",
        "utility": "[Utility:4]",
        "groundness": "[Fully supported]"
    }]
    scores = evaluate_generations(generations)
"""

import re

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tenacity import retry, wait_random_exponential
from tqdm import tqdm

# Configuration constants
LLM_AS_A_JUDGE_MODEL = "llama-3.3-70b-versatile"
FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCY_THRESHOLD = 0.7


def extract_tag_content(text: str, tag: str) -> str:
    """Extract content from specified XML tags using regex.

    Handles multi-line content and missing tags gracefully.

    Args:
        text: Input text containing XML tags
        tag: Target tag name to extract content from

    Returns:
        Extracted content as string, or empty string if not found

    Example:
        extract_tag_content("<answer>Earth</answer>", "answer") -> "Earth"
    """
    # DOTALL flag allows matching across newlines
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def evaluate_xml_structure(generation: str) -> dict[str, float]:
    """Evaluate XML structural integrity by checking required tags.

    Required tags: think, answer, contextual-relevance, answer-utility, groundness.
    Scores 1.0 per tag when exactly one opening/closing pair exists.

    Args:
        generation: XML-formatted generation text

    Returns:
        Dictionary containing:
        - Individual tag structure scores
        - Overall structure score (average of tag scores)

    Example:
        evaluate_xml_structure("<think>...</think><answer>...</answer>")
        -> {'think_structure': 1.0, ..., 'overall_structure_score': 0.8}
    """
    required_tags = ["think", "answer", "contextual-relevance", "answer-utility", "groundness"]
    scores = {}

    for tag in required_tags:
        # Count opening and closing tags
        open_count = generation.count(f"<{tag}>")
        close_count = generation.count(f"</{tag}>")

        # Valid if exactly one pair exists
        scores[f"{tag}_structure"] = 1.0 if open_count == 1 and close_count == 1 else 0.0

    # Calculate overall structural integrity score
    scores["overall_structure_score"] = sum(scores.values()) / len(required_tags)
    return scores


def extract_utility_value(utility_str: str) -> int:
    """Extract numerical value from utility token string.

    Utility tokens follow format: [Utility:<N>] where N=1-5

    Args:
        utility_str: String containing utility token

    Returns:
        Extracted integer value (1-5), or 0 if not found

    Example:
        extract_utility_value("This is [Utility:3]") -> 3
    """
    match = re.search(r"\[Utility:(\d+)\]", utility_str)
    value = int(match.group(1)) if match else 0
    return value


def evaluate_utility(extracted: str, expected: str) -> float:
    """Evaluate utility rating based on numerical comparison.

    Scoring:
    - 1.0: Exact match
    - 0.75: 1-point difference
    - 0.5: 2-point difference
    - 0.25: 3-point difference
    - 0.0: 4-point difference or invalid

    Args:
        extracted: Utility token from generation
        expected: Ground truth utility token

    Returns:
        Normalized similarity score (0.0-1.0)
    """
    if not extracted or not expected:
        return 0.0

    extracted_val = extract_utility_value(extracted)
    expected_val = extract_utility_value(expected)

    # Handle invalid extractions
    if extracted_val == 0:
        return 0.0

    # Calculate absolute difference (max possible: 4)
    diff = abs(extracted_val - expected_val)
    # Linearly scale difference to 0-1 range (higher diff = lower score)
    score = max(0.0, 1.0 - (diff / 4.0))
    return score


def evaluate_relevance(extracted: str, expected: str) -> float:
    """Evaluate relevance with exact token matching.

    Valid tokens: [Relevant], [Irrelevant]

    Args:
        extracted: Relevance token from generation
        expected: Ground truth relevance token

    Returns:
        1.0 for exact match, 0.0 otherwise
    """
    score = 0.0
    if not extracted or not expected:
        score = 0.0
    else:
        score = 1.0 if extracted.strip() == expected.strip() else 0.0
    return score


def evaluate_groundness(extracted: str, expected: str) -> float:
    """Evaluate groundness with exact token matching.

    Valid tokens:
    - [Fully supported]
    - [Partially supported]
    - [No support/Contradictory]

    Args:
        extracted: Groundness token from generation
        expected: Ground truth groundness token

    Returns:
        1.0 for exact match, 0.0 otherwise
    """
    score = 0.0
    if not extracted or not expected:
        score = 0.0
    else:
        score = 1.0 if extracted.strip() == expected.strip() else 0.0
    return score


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def evaluate_answer_correctness(question: str, answer: str, expected_answer: str) -> float:
    """Evaluate factual correctness using LLM-as-judge.

    Leverages DeepEval's GEval with custom criteria. Implements exponential backoff for reliability.

    Args:
        question: Original user question
        answer: Generated answer from model
        expected_answer: Ground truth answer

    Returns:
        Correctness score (0.0-1.0)
    """
    test_case = LLMTestCase(input=question, actual_output=answer, expected_output=expected_answer)

    # Configure evaluation criteria
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


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def evaluate_faithfulness(question: str, answer: str, context: str) -> float:
    """Evaluate faithfulness of answer to provided context.

    Uses DeepEval's FaithfulnessMetric. Implements exponential backoff.

    Args:
        question: Original user question
        answer: Generated answer from model
        context: Source context used for generation

    Returns:
        Faithfulness score (0.0-1.0)
    """
    test_case = LLMTestCase(input=question, actual_output=answer, retrieval_context=[context])

    faithfulness_metric = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD, model=LLM_AS_A_JUDGE_MODEL, include_reason=True
    )
    faithfulness_metric.measure(test_case)
    score = faithfulness_metric.score
    return score


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """Evaluate relevance of answer to original question.

    Uses DeepEval's AnswerRelevancyMetric. Implements exponential backoff.

    Args:
        question: Original user question
        answer: Generated answer from model

    Returns:
        Relevancy score (0.0-1.0)
    """
    test_case = LLMTestCase(input=question, actual_output=answer)

    relevancy_metric = AnswerRelevancyMetric(
        threshold=ANSWER_RELEVANCY_THRESHOLD, model=LLM_AS_A_JUDGE_MODEL, include_reason=True
    )
    relevancy_metric.measure(test_case)
    score = relevancy_metric.score
    return score


def evaluate_generation(data: dict) -> dict:
    """Run full evaluation pipeline on a single generation.

    Evaluation steps:
    1. XML structure validation
    2. Content extraction from XML tags
    3. Token-based correctness evaluation
    4. LLM-as-judge metrics evaluation
    5. Composite scoring

    Args:
        data: Dictionary containing:
            - instruction: User question
            - context: Source context
            - generation: XML-formatted output
            - relevance: Expected relevance token
            - utility: Expected utility token
            - groundness: Expected groundness token

    Returns:
        Dictionary containing all evaluation metrics
    """
    generation = data.get("generation", "")
    metrics = {}

    # Structural Evaluation
    structure_metrics = evaluate_xml_structure(generation)
    metrics.update(structure_metrics)

    # Content Extraction
    extracted = {
        "relevance": extract_tag_content(generation, "contextual-relevance"),
        "utility": extract_tag_content(generation, "answer-utility"),
        "groundness": extract_tag_content(generation, "groundness"),
        "answer": extract_tag_content(generation, "answer"),
    }

    # Token Correctness Evaluation
    metrics["relevance_correct"] = evaluate_relevance(extracted["relevance"], data.get("relevance"))
    metrics["utility_correct"] = evaluate_utility(extracted["utility"], data.get("utility"))
    metrics["groundness_correct"] = evaluate_groundness(extracted["groundness"], data.get("groundness"))

    # LLM-as-Judge Evaluations
    metrics["answer_correctness"] = evaluate_answer_correctness(
        question=data["instruction"],
        answer=extracted["answer"],
        expected_answer=data.get("expected_answer", extracted["answer"]),
    )

    metrics["faithfulness"] = evaluate_faithfulness(
        question=data["instruction"], answer=extracted["answer"], context=data.get("context", "")
    )

    metrics["answer_relevancy"] = evaluate_answer_relevancy(question=data["instruction"], answer=extracted["answer"])

    # Composite Scoring
    content_keys = ["relevance_correct", "utility_correct", "groundness_correct", "answer_correctness"]
    metrics["content_score"] = sum(metrics[k] for k in content_keys) / len(content_keys)

    # Final overall score averages structure and content
    metrics["overall_score"] = (metrics["overall_structure_score"] + metrics["content_score"]) / 2

    return metrics


def evaluate_generations(generations: list[dict]) -> dict[str, float]:
    """Evaluate a batch of generations and returns aggregate scores.

    Processes each generation with tqdm progress bar. Calculates average scores.

    Args:
        generations: List of data dictionaries for evaluation

    Returns:
        Dictionary of average scores across all metrics
    """
    # Initialize score accumulator
    score_accumulator = dict.fromkeys(
        [
            "think_structure",
            "answer_structure",
            "contextual-relevance_structure",
            "answer-utility_structure",
            "groundness_structure",
            "overall_structure_score",
            "relevance_correct",
            "utility_correct",
            "groundness_correct",
            "answer_correctness",
            "faithfulness",
            "answer_relevancy",
            "content_score",
            "overall_score",
        ],
        0.0,
    )

    # Process each generation with progress tracking
    for idx, data in enumerate(tqdm(generations, desc="Evaluating")):
        print(f"Evaluating example-{idx}...")
        metrics = evaluate_generation(data)

        # Aggregate scores
        for metric in score_accumulator.keys():
            score_accumulator[metric] += metrics.get(metric, 0)

    # Calculate averages and round to 2 decimals
    return {metric: round(total / len(generations), 2) for metric, total in score_accumulator.items()}
