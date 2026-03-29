from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from src.site_safety_vision.alerts import AlertGenerator
from src.site_safety_vision.config import AppConfig, load_app_config, resolve_repo_path
from src.site_safety_vision.detector import Detector
from src.site_safety_vision.matcher import PPEMatcher
from src.site_safety_vision.rules import SafetyRulesEngine
from src.site_safety_vision.utils import (
    build_output_path,
    ensure_dir,
    is_image_file,
    is_video_file,
    save_json,
    validate_file_path,
)
from src.site_safety_vision.visualization import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Site Safety Vision inference on an image or video."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or video.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference configuration YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Force saving annotated video output.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Force saving structured JSON output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )

    return parser.parse_args()


def build_pipeline(config: AppConfig) -> Dict[str, Any]:
    detector = Detector(
        model_path=config.model.model_path,
        confidence_threshold=config.model.confidence_threshold,
        device=config.model.device,
        image_size=config.model.image_size,
    )

    matcher = PPEMatcher(
        helmet_min_overlap=config.matcher.helmet_min_overlap,
        vest_min_overlap=config.matcher.vest_min_overlap,
        gloves_min_overlap=config.matcher.gloves_min_overlap,
        boots_min_overlap=config.matcher.boots_min_overlap,
    )

    rules_engine = SafetyRulesEngine(
        required_missing_frames=config.rules.required_missing_frames,
        recent_memory_frames=config.rules.recent_memory_frames,
    )

    alert_generator = AlertGenerator()
    visualizer = Visualizer()

    return {
        "detector": detector,
        "matcher": matcher,
        "rules_engine": rules_engine,
        "alert_generator": alert_generator,
        "visualizer": visualizer,
    }


def process_image(
    input_path: Path,
    output_dir: Path,
    pipeline: Dict[str, Any],
    config: AppConfig,
    save_json_enabled: bool,
) -> None:
    detector: Detector = pipeline["detector"]
    matcher: PPEMatcher = pipeline["matcher"]
    rules_engine: SafetyRulesEngine = pipeline["rules_engine"]
    alert_generator: AlertGenerator = pipeline["alert_generator"]
    visualizer: Visualizer = pipeline["visualizer"]

    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    detections = detector.track_frame(
        image,
        tracker_config=config.model.tracker,
        persist=False,
    )
    matched_results = matcher.match(detections)
    worker_states = rules_engine.evaluate_frame(matched_results)
    alerts = alert_generator.generate(worker_states)

    annotated = visualizer.annotate_frame(
        frame=image,
        detections=detections,
        matched_results=matched_results,
        worker_states=worker_states,
        alerts=alerts,
    )

    output_image_path = build_output_path(
        output_dir=output_dir,
        input_path=input_path,
        suffix="annotated",
        extension=input_path.suffix,
    )
    cv2.imwrite(str(output_image_path), annotated)
    print(f"Annotated image saved to: {output_image_path}")

    if save_json_enabled:
        output_json_path = build_output_path(
            output_dir=output_dir,
            input_path=input_path,
            suffix="results",
            extension=".json",
        )
        save_json(
            {
                "input_file": str(input_path),
                "detections": detections,
                "matched_results": matched_results,
                "worker_states": worker_states,
                "alerts": alerts,
            },
            output_json_path,
        )
        print(f"Structured results saved to: {output_json_path}")


def process_video(
    input_path: Path,
    output_dir: Path,
    pipeline: Dict[str, Any],
    config: AppConfig,
    save_video_enabled: bool,
    save_json_enabled: bool,
    max_frames: Optional[int],
) -> None:
    detector: Detector = pipeline["detector"]
    matcher: PPEMatcher = pipeline["matcher"]
    rules_engine: SafetyRulesEngine = pipeline["rules_engine"]
    alert_generator: AlertGenerator = pipeline["alert_generator"]
    visualizer: Visualizer = pipeline["visualizer"]

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    output_video_path = build_output_path(
        output_dir=output_dir,
        input_path=input_path,
        suffix="annotated",
        extension=".mp4",
    )

    if save_video_enabled:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    all_frame_results: List[Dict[str, Any]] = []
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.track_frame(
                frame,
                tracker_config=config.model.tracker,
                persist=True,
            )
            matched_results = matcher.match(detections)
            worker_states = rules_engine.evaluate_frame(matched_results)
            alerts = alert_generator.generate(worker_states)

            annotated = visualizer.annotate_frame(
                frame=frame,
                detections=detections,
                matched_results=matched_results,
                worker_states=worker_states,
                alerts=alerts,
            )

            if writer is not None:
                writer.write(annotated)

            if save_json_enabled:
                all_frame_results.append(
                    {
                        "frame_index": frame_index,
                        "detections": detections,
                        "matched_results": matched_results,
                        "worker_states": worker_states,
                        "alerts": alerts,
                    }
                )

            if frame_index % 30 == 0:
                print(f"Processed frame {frame_index}")

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if save_video_enabled:
        print(f"Annotated video saved to: {output_video_path}")

    if save_json_enabled:
        output_json_path = build_output_path(
            output_dir=output_dir,
            input_path=input_path,
            suffix="results",
            extension=".json",
        )
        save_json(
            {
                "input_file": str(input_path),
                "total_frames_processed": frame_index,
                "frames": all_frame_results,
            },
            output_json_path,
        )
        print(f"Structured results saved to: {output_json_path}")


def main() -> None:
    args = parse_args()

    input_path = validate_file_path(args.input, must_exist=True)
    config_path = validate_file_path(args.config, must_exist=True)

    config = load_app_config(config_path)

    if args.output_dir:
        output_dir = ensure_dir(args.output_dir)
    else:
        output_dir = ensure_dir(config.output.output_dir)

    save_video_enabled = args.save_video or config.output.save_video
    save_json_enabled = args.save_json or config.output.save_json

    pipeline = build_pipeline(config)

    if is_video_file(input_path):
        process_video(
            input_path=input_path,
            output_dir=output_dir,
            pipeline=pipeline,
            config=config,
            save_video_enabled=save_video_enabled,
            save_json_enabled=save_json_enabled,
            max_frames=args.max_frames,
        )
    elif is_image_file(input_path):
        process_image(
            input_path=input_path,
            output_dir=output_dir,
            pipeline=pipeline,
            config=config,
            save_json_enabled=save_json_enabled,
        )
    else:
        raise ValueError(
            f"Unsupported file type: {input_path.suffix}. "
            "Please provide a supported image or video file."
        )


if __name__ == "__main__":
    main()