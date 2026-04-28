"""
main.py — Orchestrator for the video-only deception detection pipeline.
"""

import argparse
import logging
import os
import traceback
from typing import List

from tqdm import tqdm

import config
from data_loader import Sample, load_dataset, validate_dataset
from evaluate import compute_metrics, print_summary, save_metrics, save_predictions
from frame_sampler import sample_frames
from model_inference import run_inference
from output_parser import ParsedOutput, parse_output
from prompt_builder import build_prompt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video-only deception detection with Qwen3-VL-8B"
    )
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=config.DEFAULT_MODEL)
    parser.add_argument("--num-frames", type=int, default=config.NUM_FRAMES)
    parser.add_argument("--output-dir", type=str, default=config.RESULTS_DIR)
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    logger.info("Model : %s", args.model_id)
    logger.info("Frames: %d per video", args.num_frames)

    predictions_path = os.path.join(args.output_dir, "predictions.json")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)

    samples: List[Sample] = load_dataset()
    validate_dataset(samples)

    if args.subset:
        samples = samples[: args.subset]
        logger.info("Subset mode: running on %d samples.", len(samples))

    prompt = build_prompt()

    sample_ids = []
    ground_truths = []
    predictions: List[ParsedOutput] = []

    for sample in tqdm(samples, desc="Running inference", unit="clip"):
        sample_ids.append(sample.sample_id)
        ground_truths.append(sample.label_int)

        try:
            frames = sample_frames(sample.video_path, num_frames=args.num_frames)
            raw_output = run_inference(frames, prompt, model_id=args.model_id)
            prediction = parse_output(raw_output)
        except Exception as exc:
            logger.error(
                "Error on sample %s: %s\n%s",
                sample.sample_id,
                exc,
                traceback.format_exc(),
            )
            prediction = ParsedOutput(
                predicted_label="unknown",
                predicted_int=-1,
                confidence=0.5,
                raw_output=f"[ERROR: {exc}]",
                parse_status="failed",
            )

        predictions.append(prediction)

        correct = "OK" if prediction.predicted_int == sample.label_int else "ERR"
        tqdm.write(
            f"  {sample.sample_id}  gt={sample.label:<10}  "
            f"pred={prediction.predicted_label:<10}  "
            f"conf={prediction.confidence:.2f}  "
            f"[{prediction.parse_status}]  {correct}"
        )

    metrics = compute_metrics(ground_truths, predictions)
    print_summary(metrics)
    save_predictions(sample_ids, ground_truths, predictions, path=predictions_path)
    save_metrics(metrics, path=metrics_path)
    logger.info("Done. Results written to %s", args.output_dir)


if __name__ == "__main__":
    run_pipeline(parse_args())