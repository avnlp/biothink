"""Create stratified subsets for training, validation, and testing on the Self-BioRAG dataset.

This module provides functionality to create balanced train, validation, and test splits
from the processed Self-BioRAG dataset. It ensures class balance by stratifying the splits
based on the relevance labels, and applies quality filters to maintain data integrity.

The script performs the following operations:
1. Loads the processed Self-BioRAG dataset from the Hugging Face Hub
2. Applies quality filters to remove incomplete or low-quality examples
3. Creates stratified splits for training, validation, and test sets
4. Maintains class balance across all splits based on relevance labels
5. Pushes the processed subset back to the Hugging Face Hub
"""

from datasets import DatasetDict, load_dataset

DATASET_NAME = "avnlp/self_biorag_processed"
TRAIN_SIZE = 100
TEST_SIZE = 40
VALIDATION_SIZE = 20
MINIMUM_CONTEXT_LENGTH = 300
STRATIFY_COLUMN = "relevance"
STRATIFY_COLUMN_CLASS_LABEL = f"{STRATIFY_COLUMN}_class_label"

dataset = load_dataset(DATASET_NAME, split="train", download_mode="force_redownload")

# Filter out examples without context
dataset = dataset.filter(lambda x: x["context"] is not None)

# Filter out examples with context length less than MINIMUM_CONTEXT_LENGTH
dataset = dataset.filter(lambda x: len(x["context"]) > MINIMUM_CONTEXT_LENGTH)

# Filter out examples without relevance tokens
dataset = dataset.filter(lambda x: x["relevance"] is not None)

# Filter out examples without answer
dataset = dataset.filter(lambda x: x["answer"] is not None)

# Stratification column has to be of type ClassLabel, so we add copy of relevance column to use for stratification
dataset = dataset.map(lambda x: {STRATIFY_COLUMN_CLASS_LABEL: x[STRATIFY_COLUMN]})
dataset = dataset.class_encode_column(column=STRATIFY_COLUMN_CLASS_LABEL)

# Create Train, Test, and Validation splits
dataset = dataset.train_test_split(
    train_size=TRAIN_SIZE, test_size=TEST_SIZE, stratify_by_column=STRATIFY_COLUMN_CLASS_LABEL
)

# Split the test dataset further into test and validation
test_dataset = dataset["test"].train_test_split(
    train_size=VALIDATION_SIZE, test_size=VALIDATION_SIZE, stratify_by_column=STRATIFY_COLUMN_CLASS_LABEL
)

subset_dataset = DatasetDict(
    {
        "train": dataset["train"],
        "test": test_dataset["train"],
        "validation": test_dataset["test"],
    }
)

# Remove additional columns
subset_dataset = subset_dataset.remove_columns(STRATIFY_COLUMN_CLASS_LABEL)

# Push to HuggingFace Hub
subset_dataset.push_to_hub(
    DATASET_NAME + "_subset",
    private=False,
)
