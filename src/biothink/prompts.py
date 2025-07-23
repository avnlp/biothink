"""Prompts for BioThink training."""

RELEVANCE_PROMPT = """
Relevance:
- [Relevant]: Context directly pertains to the question and aids in answering.
- [Irrelevant]: Context does not help or is off-topic.
"""

UTILITY_PROMPT = """
Utility (1-5 scale):
[Utility:5]: Complete, highly detailed, fully satisfies the query.
[Utility:4]: Mostly satisfies; minor improvements possible (structure, depth).
[Utility:3]: Adequate but needs major additions or clarifications.
[Utility:2]: Partially addresses the request; significant gaps remain.
[Utility:1]: Barely on-topic or off-topic.
"""

GROUNDNESS_PROMPT = """
Groundness:
- [Fully supported]: Every claim in the answer is backed by the context.
- [Partially supported]: Some claims go beyond or are missing from the context.
- [No support/Contradictory]: Answer ignores or contradicts the context.
"""


SYSTEM_PROMPT = f"""You are given a <Question>, its <Context>.
1. Think through the question and context; show your reasoning within <think>…</think>.
2. Provide your final answer within <answer>…</answer>.
3. Assess three dimensions—Relevance, Utility, and Groundness—using the criteria below.

{RELEVANCE_PROMPT}
{UTILITY_PROMPT}
{GROUNDNESS_PROMPT}

Respond exactly in this format:

<think>
…your step-by-step reasoning…
</think>

<answer>
…your concise answer…
</answer>

<contextual-relevance>
[Relevant] or [Irrelevant]
</contextual-relevance>

<answer-utility>
[Utility:5] or [Utility:4] or [Utility:3] or [Utility:2] or [Utility:1]
</answer-utility>

<groundness>
[Fully supported] or [Partially supported] or [No support/Contradictory]
</groundness>
"""

USER_PROMPT = """Answer the question: {question} using the context: {context}"""
