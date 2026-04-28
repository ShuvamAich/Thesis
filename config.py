"""
config.py — Central configuration for the video-only deception detection pipeline.
"""

import os


DATASET_ROOT = r"c:\Thesis\Real-life_Deception_Detection_2016"

CLIPS_DECEPTIVE_DIR = os.path.join(DATASET_ROOT, "Clips", "Deceptive")
CLIPS_TRUTHFUL_DIR = os.path.join(DATASET_ROOT, "Clips", "Truthful")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "predictions.json")
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.json")

MODEL_QWEN3_VL_8B = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL = MODEL_QWEN3_VL_8B

MAX_NEW_TOKENS = 64
TEMPERATURE = 0.1
DO_SAMPLE = False

NUM_FRAMES = 8

LABEL_MAP = {
    "deceptive": 1,
    "truthful": 0,
}
LABEL_NAMES = {value: key for key, value in LABEL_MAP.items()}