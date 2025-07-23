"""Evaluate Qwen3 generations."""

import json

from biothink.evaluation.metrics import evaluate_generations

# Load generations
with open("src/biothink/generation/generations_qwen3_1.7b.jsonl") as f:
    generations = [json.loads(line) for line in f]

# Evaluate generations
scores = evaluate_generations(generations)
print(scores)

# Save scores
with open("src/biothink/evaluation/scores_qwen3_1.7b.json", "w") as f:
    json.dump(scores, f)
