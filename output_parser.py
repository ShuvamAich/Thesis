"""
output_parser.py — Extract structured prediction from raw model output text.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_LABEL_PATTERN = re.compile(
    r"final\s+label\s*:\s*(truthful|deceptive)"
    r"(?:\s*\(confidence\s*:\s*([0-9]*\.?[0-9]+)\))?",
    re.IGNORECASE,
)
_WORD_ONLY_PATTERN = re.compile(r"^\s*(truthful|deceptive)\s*$", re.IGNORECASE)
_FALLBACK_PATTERN = re.compile(r"\b(truthful|deceptive)\b", re.IGNORECASE)


@dataclass
class ParsedOutput:
    predicted_label: str
    predicted_int: int
    confidence: float
    raw_output: str
    parse_status: str


def parse_output(raw_output: str) -> ParsedOutput:
    match = _LABEL_PATTERN.search(raw_output)
    if match:
        label = match.group(1).lower()
        conf_str = match.group(2)
        confidence = float(conf_str) if conf_str is not None else 0.5
        confidence = max(0.0, min(1.0, confidence))
        return ParsedOutput(
            predicted_label=label,
            predicted_int=1 if label == "deceptive" else 0,
            confidence=confidence,
            raw_output=raw_output,
            parse_status="ok",
        )

    word_only_match = _WORD_ONLY_PATTERN.match(raw_output)
    if word_only_match:
        label = word_only_match.group(1).lower()
        return ParsedOutput(
            predicted_label=label,
            predicted_int=1 if label == "deceptive" else 0,
            confidence=0.5,
            raw_output=raw_output,
            parse_status="ok",
        )

    mentions = list(_FALLBACK_PATTERN.finditer(raw_output))
    if mentions:
        label = mentions[-1].group(1).lower()
        logger.warning("Primary label pattern not found; using fallback label '%s'.", label)
        return ParsedOutput(
            predicted_label=label,
            predicted_int=1 if label == "deceptive" else 0,
            confidence=0.5,
            raw_output=raw_output,
            parse_status="fallback",
        )

    logger.error("Could not extract a label from model output:\n%s", raw_output[:300])
    return ParsedOutput(
        predicted_label="unknown",
        predicted_int=-1,
        confidence=0.5,
        raw_output=raw_output,
        parse_status="failed",
    )