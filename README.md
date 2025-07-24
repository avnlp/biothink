# BioThink: Self-Reflective Reasoning for Biomedical QA

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/avnlp/BioThink-Qwen3-1.7B)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/avnlp/self_biorag_processed)

## Introduction

Recent Large Language Models (LLMs) have achieved remarkable success in a wide range of tasks, including question answering, text generation, and reasoning. However, these LLMs often struggle with domain-specific tasks, such as biomedical question answering, without extensive pre-training on domain-specific data.

Inspired by [Self-RAG](https://arxiv.org/abs/2310.11511) and building upon [Self-BioRAG](https://arxiv.org/abs/2401.15269), we introduce **BioThink**, a framework that enhances LLMs for biomedical question answering through self-reflection, context grading, relevance assessment, and utility rating. BioThink uses a novel training approach with GRPO (Group Relative Policy Optimization) to fine-tune LLMs to generate structured outputs that include step-by-step reasoning, concise answers, and self-reflection tokens.

## Key Features

- **Self-Reflective Generation**: BioThink generates outputs in a structured format that includes:
  - Step-by-step reasoning (`<think>`)
  - Concise answer (`<answer>`)
  - Contextual relevance assessment (`<contextual-relevance>`)
  - Answer utility rating (`<answer-utility>`)
  - Groundness evaluation (`<groundness>`)
- **Training with GRPO**: We use Group Relative Policy Optimization (GRPO) to train the model, incorporating multiple reward functions to ensure:
  - Correctness of the answer
  - Accuracy of self-reflection tokens (utility, relevance, groundness)
  - Proper XML structure and order of tags
  - Faithfulness and relevancy of the answer
- **Efficiency**: The model is trained using QLoRA and Unsloth for efficient fine-tuning.

## Training Steps

### 1. Data Processing

 The [Self-BioRAG dataset](https://github.com/dmis-lab/self-biorag) is processed using the script [process_data.py](src/biothink/process_data.py). This script extracts questions, answers, and context, and also prepares labels for groundness, relevance, and utility tokens.
 The processed dataset is available at [avnlp/self_biorag_processed](https://huggingface.co/datasets/avnlp/self_biorag_processed).

### 2. Model Training

 The model is trained using the script [train_rag.py](src/biothink/train_rag.py). The training process involves:

**Structured Generation**: The model is trained to generate outputs in the following format:

   ```xml
   <think>
   ... step-by-step reasoning ...
   </think>
   <answer>
   ... concise answer ...
   </answer>
   <contextual-relevance>
   [Relevant] or [Irrelevant]
   </contextual-relevance>
   <answer-utility>
   [Utility:5] or [Utility:4] or ... [Utility:1]
   </answer-utility>
   <groundness>
   [Fully supported] or [Partially supported] or [No support/Contradictory]
   </groundness>
   ```

**Reward Functions**: The training uses GRPO with the following rewards:

- **Correctness Reward**: Measures answer correctness using DeepEval's GEval metric with a custom LLM-as-a-Judge instruction tailored for Bio-Medical Question Answering.
- **Utility Reward**: Ensures the correct Utility token is generated.
- **Relevance Reward**: Ensures the correct Relevance token is generated.
- **Groundness Reward**: Ensures the correct Groundness token is generated.
- **XML Structure Reward**: Checks for the presence and proper opening/closing of all required tags.
- **Structure Order Reward**: Ensures the tags appear in the correct order and that no extra text is present outside the tags.

### 3. Model

We fine-tune the `Qwen-3-1.7B` model using GRPO and QLoRA. The trained model is available on Hugging Face:
[avnlp/BioThink-Qwen3-1.7B](https://huggingface.co/avnlp/BioThink-Qwen3-1.7B).

### 4. Evaluation

 The model is evaluated using the following metrics:

1. **XML Structure**: Checks for the presence of the opening and closing of all reasoning, answer, contextual-relevance, answer-utility, groundness tags.
2. **Utility**: Checks that the correct utility token has been generated.
3. **Relevance**: Checks that the correct relevance token has been generated.
4. **Groundness**: Checks that the correct groundness token has been generated.
5. **Answer Correctness**: Checks that the answer is correct using DeepEval's GEval metric with a custom instruction for LLM-as-a-Judge.
6. **Faithfulness**: Checks that the answer is faithful to the provided context using DeepEval's Faithfulness LLM-as-a-Judge metric.
7. **Answer Relevancy**: Checks that the answer is relevant to the original question using DeepEval's Answer Relevancy LLM-as-a-Judge metric.
