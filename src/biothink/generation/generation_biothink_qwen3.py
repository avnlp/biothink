"""Model Inference Pipeline with the trained BioThink Qwen3 model.

This script performs batch inference using the BioThink Qwen3 model for biomedical question answering.

The pipeline:
1. Loads a quantized model optimized with Unsloth
2. Processes a validation dataset with context-rich biomedical questions
3. Generates responses using a structured prompt template
4. Captures full input/output metadata
5. Saves results in JSON Lines format for evaluation
"""

import json

from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams

from biothink.prompts import SYSTEM_PROMPT, USER_PROMPT

# Model configuration
MODEL_NAME = "avnlp/BioThink-Qwen3-1.7B"
MAX_SEQ_LENGTH = 10240
LORA_RANK = 32
GPU_MEMORY_UTILIZATION = 1.0

# Generation parameters
TEMPERATURE = 0.6
TOP_K = 20
TOP_P = 0.95
MIN_P = 0
MAX_NEW_TOKENS = 1500

# Dataset configuration
DATASET_NAME = "avnlp/self_biorag_processed_subset"
SPLIT = "validation"

# Initialize model with optimized 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)

# Load validation dataset
dataset = load_dataset(DATASET_NAME, split=SPLIT)

# Initialize results container
results = []

# Configure generation parameters (fixed for all samples)
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    min_p=MIN_P,
    max_tokens=MAX_NEW_TOKENS,
)

print(f"Starting inference on {len(dataset)} validation examples...")

# Process each example in the dataset
for i, row in enumerate(dataset):
    # Extract question and context from dataset row
    question = row["instruction"]
    context = row["context"]

    # Format messages using predefined prompt templates
    messages = [
        # System prompt defines model behavior constraints
        {"role": "system", "content": SYSTEM_PROMPT},
        # User prompt incorporates question and context
        {"role": "user", "content": USER_PROMPT.format(question=question, context=context)},
    ]

    # Apply chat template to create model-ready input
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Generate response using optimized inference
    generation = (
        model.fast_generate(
            text,
            sampling_params=sampling_params,
        )[0]
        .outputs[0]
        .text
    )

    # Preserve original data and add inference artifacts
    result = dict(row)
    result.update(
        {
            "index": i,
            "messages": messages,
            "formatted_text": text,
            "generation": generation,
        }
    )
    results.append(result)

    # Periodic progress updates with samples
    if (i + 1) % 10 == 0 or i == 0:
        print(f"Processed {i + 1}/{len(dataset)} examples")
        print(f"Sample input: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"Sample output: {generation[:200]}{'...' if len(generation) > 200 else ''}")
        print("-" * 50)

print(f"\nCompleted inference on {len(results)} examples!")
print(f"Results contain keys: {list(results[0].keys())}")

# Save results to JSON Lines file
output_file = "generations_biothink_qwen3_1.7b.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Results saved to: {output_file}")
print(f"File contains {len(results)} records in JSONL format")
