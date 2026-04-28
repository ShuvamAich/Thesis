"""
prompt_builder.py — Assemble a strict video-only classification prompt.
"""

_PROMPT = """\
You are an expert judge, specializing in deception detection. Your task is to evaluate visual evidence from courtroom trials or interviews with exceptional accuracy.

Based on the provided sampled video frames, assess whether the speaker's behaviour appears Truthful or Deceptive.

Use only what is visible in the frames. Do not assume access to transcripts, gesture annotations from a CSV, audio, or external metadata. Utilize visible behavioural cues, contextual clues from the scene, and consistency across the sampled frames to make your judgment.

Carefully analyze the frames below, paying close attention to the visible content and context, and provide your judgment.

Respond with only one word: either Truthful or Deceptive.
Do not provide additional explanation, reasoning, commentary, punctuation, or confidence scores in your response.
"""


def build_prompt() -> str:
    return _PROMPT