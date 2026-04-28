"""
frame_sampler.py — Extract uniformly spaced frames from a video file.
"""

import os
from typing import List

import cv2
from PIL import Image

import config


def sample_frames(video_path: str, num_frames: int = config.NUM_FRAMES) -> List[Image.Image]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count from: {video_path}")

    num_frames = min(num_frames, total_frames)
    step = total_frames / num_frames
    indices = [int(step * index + step / 2) for index in range(num_frames)]

    frames: List[Image.Image] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be extracted from: {video_path}")

    return frames