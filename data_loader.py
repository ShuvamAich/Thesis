"""
data_loader.py — Load video paths and labels for the video-only pipeline.
"""

import os
from dataclasses import dataclass
from typing import List

import config


@dataclass
class Sample:
    sample_id: str
    video_path: str
    label: str
    label_int: int


def _collect_samples(directory: str, label: str) -> List[Sample]:
    samples: List[Sample] = []
    for file_name in sorted(os.listdir(directory)):
        if not file_name.lower().endswith(".mp4"):
            continue
        samples.append(
            Sample(
                sample_id=file_name,
                video_path=os.path.join(directory, file_name),
                label=label,
                label_int=config.LABEL_MAP[label],
            )
        )
    return samples


def load_dataset() -> List[Sample]:
    deceptive = _collect_samples(config.CLIPS_DECEPTIVE_DIR, "deceptive")
    truthful = _collect_samples(config.CLIPS_TRUTHFUL_DIR, "truthful")
    return deceptive + truthful


def validate_dataset(samples: List[Sample]) -> None:
    missing_videos = [sample for sample in samples if not os.path.isfile(sample.video_path)]

    print(
        f"[data_loader] Loaded {len(samples)} samples "
        f"({sum(1 for s in samples if s.label == 'deceptive')} deceptive, "
        f"{sum(1 for s in samples if s.label == 'truthful')} truthful)"
    )

    if missing_videos:
        raise RuntimeError(
            f"[data_loader] {len(missing_videos)} video file(s) not found:\n"
            + "\n".join(f"  {sample.video_path}" for sample in missing_videos[:5])
        )