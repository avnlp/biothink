"""BioThink Model Fine-Tuning Pipeline.

This script implements a full workflow for fine-tuning LLMs using Group Relative Policy Optimization (GRPO) for bio-medical reasoning question-answering.

The pipeline includes:
- Model and tokenizer initialization with optimized 4-bit quantization
- Dataset loading and prompt formatting
- Training configuration setup with GRPO parameters
- Multi-objective reward function integration
- Training execution and model saving

The fine-tuning process incorporates five reward functions that evaluate:
1. Answer correctness (using LLM-as-a-judge)
2. XML structure compliance
3. Utility rating accuracy
4. Contextual relevance
5. Groundness in source material

Key Components:
- Unsloth for accelerated LoRA fine-tuning
- vLLM for efficient generation
- TRL's GRPOTrainer for policy optimization
- Custom reward functions for domain-specific evaluation
"""

from datasets import load_dataset
from train_config import (
    DATASET_NAME,
    GPU_MEMORY_UTILIZATION,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_RANK,
    LR_SCHEDULER_TYPE,
    MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH,
    MAX_STEPS,
    MIN_P,
    MODEL_NAME,
    NUM_GENERATIONS,
    OPTIM,
    OUTPUT_DIR,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    REPORT_TO,
    SAVE_STEPS,
    SPLIT,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from vllm import SamplingParams

from biothink.prompts import SYSTEM_PROMPT, USER_PROMPT
from biothink.reward_functions import (
    correctness_reward_func,
    groundness_reward_func,
    relevance_reward_func,
    utility_reward_func,
    xml_structure_reward_func,
)

# Initialize model with 4-bit quantization and memory optimization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)

# Apply Parameter-Efficient Fine-Tuning (LoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load and preprocess training dataset
dataset = load_dataset(DATASET_NAME, split=SPLIT)

# Format prompts with system instructions and user context
dataset = dataset.map(
    lambda x: {
        "prompt": [
            # System prompt defines the AI's role and constraints
            {"role": "system", "content": SYSTEM_PROMPT},
            # User prompt incorporates question and context
            {"role": "user", "content": USER_PROMPT.format(question=x["instruction"], context=x["context"])},
        ],
    }
)

# Configure sampling parameters for response generation
vllm_sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    min_p=MIN_P,
    max_tokens=MAX_NEW_TOKENS,
)

# Set up GRPO training configuration
training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    optim=OPTIM,
    logging_steps=LOGGING_STEPS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_NEW_TOKENS,
    max_steps=MAX_STEPS,
    save_steps=SAVE_STEPS,
    report_to=REPORT_TO,
    output_dir=OUTPUT_DIR,
)

# Initialize GRPO trainer with multi-objective rewards
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        correctness_reward_func,
        xml_structure_reward_func,
        utility_reward_func,
        relevance_reward_func,
        groundness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# Execute training process
trainer.train()

# Save LoRA adapter weights
model.save_lora("grpo_saved_lora")

# Merge and save final model
model.save_pretrained_merged(
    "model",
    tokenizer,
    save_method="merged_4bit",
)

# Push merged model to Hugging Face Hub
model.push_to_hub_merged(
    "avnlp/BioThink-Qwen3-1.7B",
    tokenizer,
    save_method="merged_4bit",
)
