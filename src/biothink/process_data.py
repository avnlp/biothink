"""Process Self-BioRAG Dataset.

This module provides functionality to process and extract structured information from
Self-BioRAG dataset text outputs. The dataset contains text with specific token-based
formats that encode retrieval information, context, relevance assessments, answers,
groundness evaluations, and utility ratings.

The Self-BioRAG dataset has been taken from: https://github.com/dmis-lab/self-biorag.

Data Format:
    The text follows this pattern:
    [Retrieval]<paragraph>context_text</paragraph>[Relevant/Irrelevant]answer_text[groundness_token][utility_token]

Example:
    "[Retrieval]<paragraph>currently matter of discussion, called passive symplasmic loading.</paragraph>
    [Relevant]The final answer is (B).[Fully supported][Utility:4]"

Token Categories:
    - Retrieval Control: "[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"
    - Relevance Assessment: "[Relevant]", "[Irrelevant]"
    - Groundness Evaluation: "[No support / Contradictory]", "[Fully supported]", "[Partially supported]"
    - Utility Rating: "[Utility:1]" through "[Utility:5]"

Processing Rules:
    1. Rows containing "[No Retrieval]" or "[Continue to Use Evidence]" are filtered out
    2. Only rows beginning with "[Retrieval]" are processed
    3. Context is extracted from <paragraph></paragraph> XML-like tags
    4. Answer text is extracted between relevance and special tokens
    5. For "[Relevant]" cases: groundness tokens precede utility tokens
    6. For "[Irrelevant]" cases: only utility tokens are expected (no groundness)

Output Columns:
    - context: Text content from paragraph tags
    - relevance: Relevance assessment token
    - answer: Extracted answer text
    - groundness: Groundness evaluation (relevant cases only)
    - utility: Utility rating token
"""

import re

from datasets import load_dataset


def format_output_for_dataset(example):
    """Extract structured information from a single dataset row's output text.

    Extract context, relevance assessments, answers, groundness evaluations, and utility ratings from the output text.

    The example format is:
    [Retrieval]<paragraph>context_text</paragraph>[Relevant/Irrelevant]answer_text[groundness_token][utility_token]

    Processing Rules:
    - Assumes input has already been filtered to exclude unwanted tokens
    - Only processes rows starting with "[Retrieval]"
    - Context is extracted from <paragraph></paragraph> XML-like tags
    - Relevance must be either "[Relevant]" or "[Irrelevant]"
    - For "[Relevant]" cases: groundness tokens are expected before utility tokens
    - For "[Irrelevant]" cases: no groundness tokens are expected, only utility tokens
    - Answer text is everything between relevance token and the next special token

    Args:
        example (dict): A single row from the dataset containing at least an 'output' key
            with the text to be processed. The dictionary represents one sample from
            a HuggingFace Dataset.

    Returns:
        dict: Original example data plus extracted fields. Always returns a valid dict.

            Extracted fields added to the returned dict:
            - context (str or None): Text content from <paragraph></paragraph> tags
            - relevance (str or None): "[Relevant]" or "[Irrelevant]" token
            - answer (str or None): Answer text between relevance and groundness/utility tokens
            - groundness (str or None): One of "[No support / Contradictory]",
              "[Fully supported]", "[Partially supported]" (only for relevant cases)
            - utility (str or None): One of "[Utility:1]" through "[Utility:5]"

    Example:
        >>> example = {"output": "[Retrieval]<paragraph>Some context</paragraph>[Relevant]Answer text[Fully supported][Utility:3]"}
        >>> result = format_output_for_dataset(example)
        >>> print(result["context"])
        >>> print(result["answer"])
    """
    # Extract the output text from the example
    output_text = example.get("output", "")

    # Initialize the result dictionary with all original example data to preserve existing columns
    result = dict(example)

    # Initialize extracted fields with None values (default state)
    # These will be populated if the corresponding tokens/content are found
    extracted_fields = {
        "context": None,
        "relevance": None,
        "answer": None,
        "groundness": None,
        "utility": None,
    }

    # Only process rows that start with [Retrieval] token
    if output_text.startswith("[Retrieval]"):

        # Extract context from <paragraph></paragraph> XML-like tags, match text including newlines
        context_match = re.search(r"<paragraph>(.*?)</paragraph>", output_text, re.DOTALL)
        if context_match:
            # Extract and clean the context text
            extracted_fields["context"] = context_match.group(1).strip()
            # Set remaining text to everything after the closing </paragraph> tag
            remaining = output_text[context_match.end() :].strip()
        else:
            # No paragraph tags found, start processing after [Retrieval] token
            remaining = output_text[len("[Retrieval]") :].strip()

        # Extract relevance token ([Relevant] or [Irrelevant])
        # Check which relevance token appears at the start of remaining text
        relevance_tokens = ["[Relevant]", "[Irrelevant]"]
        for token in relevance_tokens:
            if remaining.startswith(token):
                extracted_fields["relevance"] = token
                # Update remaining text by removing the found relevance token
                remaining = remaining[len(token) :].strip()
                break  # Stop after finding the first matching token

        # Define all possible special tokens that can appear after the answer
        groundness_tokens = ["[No support / Contradictory]", "[Fully supported]", "[Partially supported]"]
        utility_tokens = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
        all_tokens = groundness_tokens + utility_tokens

        # Find the position of the earliest occurring special token
        # This determines where the answer text ends
        next_token_pos = None
        next_token = None
        for token in all_tokens:
            pos = remaining.find(token)
            # Update if this token appears earlier than previously found tokens
            if pos != -1 and (next_token_pos is None or pos < next_token_pos):
                next_token_pos = pos
                next_token = token

        # Extract answer text and process remaining special tokens
        if next_token_pos is not None:
            # Extract answer as text before the first special token
            extracted_fields["answer"] = remaining[:next_token_pos].strip()
            # Get text remaining after the found special token
            remaining_after_answer = remaining[next_token_pos + len(next_token) :].strip()

            # Process tokens based on relevance type and token category
            # Different processing logic for relevant vs irrelevant cases
            if extracted_fields["relevance"] == "[Relevant]":
                # For relevant cases, both groundness and utility tokens are expected
                if next_token in groundness_tokens:
                    # Found groundness token first - extract it
                    extracted_fields["groundness"] = next_token
                    # Look for utility token in the remaining text after groundness
                    for util_token in utility_tokens:
                        if util_token in remaining_after_answer:
                            extracted_fields["utility"] = util_token
                            break  # Stop after finding first utility token
                elif next_token in utility_tokens:
                    # Found utility token first (no groundness token present)
                    extracted_fields["utility"] = next_token

            elif extracted_fields["relevance"] == "[Irrelevant]":
                # For irrelevant cases, only utility tokens are expected (no groundness)
                if next_token in utility_tokens:
                    extracted_fields["utility"] = next_token
        else:
            # No special tokens found - everything remaining is the answer
            # This handles cases where the text ends with the answer
            extracted_fields["answer"] = remaining.strip()

    # Merge extracted fields into the result dictionary, add the new columns while preserving original data
    result.update(extracted_fields)

    return result


def process_dataset(dataset, remove_original_output=False):
    """Process a HuggingFace dataset to extract structured information from text outputs.

    This function processes a dataset containing text outputs with specific token patterns
    and extracts structured information into separate columns. It performs the following
    operations:

    1. Removes all rows containing "[No Retrieval]" or "[Continue to Use Evidence]" tokens
    2. Processes only rows that begin with "[Retrieval]" token
    3. Extracts context from <paragraph></paragraph> tags
    4. Extracts relevance tokens ("[Relevant]" or "[Irrelevant]")
    5. Extracts answer text (content between relevance and groundness/utility tokens)
    6. Extracts groundness tokens ("[No support / Contradictory]", "[Fully supported]",
       "[Partially supported]") - only for relevant cases
    7. Extracts utility tokens ("[Utility:1]" through "[Utility:5]")

    Args:
        dataset (datasets.Dataset): HuggingFace Dataset object containing an 'output' column
            with text to be processed. Each row should contain structured text with the
            token patterns described above.
        remove_original_output (bool, optional): If True, removes the original 'output'
            column from the processed dataset to save memory. Defaults to False.

    Returns:
        datasets.Dataset: Processed dataset with the following new columns:
            - context (str or None): Text extracted from <paragraph></paragraph> tags
            - relevance (str or None): Relevance token ("[Relevant]" or "[Irrelevant]")
            - answer (str or None): Answer text extracted after relevance token
            - groundness (str or None): Groundness assessment token (only for relevant cases)
            - utility (str or None): Utility rating token ("[Utility:1]" through "[Utility:5]")

            Rows that don't match the expected pattern or contain skip tokens are filtered out.

    Example:
        >>> dataset = load_dataset("path/to/dataset")
        >>> processed = process_dataset(dataset, remove_original_output=True)
        >>> print(processed.column_names)
        ['context', 'relevance', 'answer', 'groundness', 'utility']
    """
    # First, filter out rows with unwanted tokens to avoid multiprocessing issues
    print("Filtering dataset to remove unwanted tokens...")
    filtered = dataset.filter(
        lambda x: "[No Retrieval]" not in x.get("output", "")
        and "[Continue to Use Evidence]" not in x.get("output", "")
        and x.get("output", "").startswith("[Retrieval]"),
        num_proc=4,
    )

    print(f"Dataset size after filtering: {len(filtered)} (was {len(dataset)})")

    # Apply the format function to each row in the filtered dataset using multiprocessing
    print("Processing dataset to extract structured information...")
    processed = filtered.map(format_output_for_dataset, batched=False, num_proc=4)

    # Optionally remove the original output column
    if remove_original_output and "output" in processed.column_names:
        processed = processed.remove_columns(["output"])

    return processed


# Load dataset
DATASET_NAME = "avnlp/self_biorag"
dataset = load_dataset(DATASET_NAME, split="train", download_mode="force_redownload")

# Process dataset
processed_dataset = process_dataset(dataset)

# Push to HuggingFace Hub
processed_dataset.push_to_hub(
    DATASET_NAME + "_processed",
    split="train",
    private=False,
)
