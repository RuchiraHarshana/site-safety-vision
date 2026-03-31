from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2

from site_safety_vision.alerts import AlertGenerator
from site_safety_vision.analytics import SiteSafetyAnalyticsEngine
from site_safety_vision.config import AppConfig, load_app_config, resolve_repo_path
from site_safety_vision.detector import Detector
from site_safety_vision.matcher import PPEMatcher
from site_safety_vision.review import IncidentReviewEngine
from site_safety_vision.rules import SafetyRulesEngine
from site_safety_vision.visualization import Visualizer


# ----------------------------
# Path / file helpers
# ----------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def validate_video_path(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Path is not a file: {video_path}")
    if not is_video_file(video_path):
        raise ValueError(
            f"Unsupported video file extension: {video_path.suffix}. "
            f"Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )


def build_output_path(output_dir: Path, input_path: Path, suffix: str, extension: str) -> Path:
    stem = input_path.stem
    return output_dir / f"{stem}_{suffix}{extension}"


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert values into JSON-safe types.
    """
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


# ----------------------------
# Config / pipeline helpers
# ----------------------------

def resolve_config_path(config_input: str) -> Path:
    candidate = Path(config_input).expanduser()

    if candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate.exists():
        return candidate.resolve()

    repo_candidate = resolve_repo_path(config_input)
    if repo_candidate.exists():
        return repo_candidate.resolve()

    raise FileNotFoundError(f"Config file not found: {config_input}")


def resolve_model_path(model_path: str) -> str:
    candidate = Path(model_path).expanduser()

    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    if candidate.exists():
        return str(candidate.resolve())

    repo_candidate = resolve_repo_path(model_path)
    if repo_candidate.exists():
        return str(repo_candidate.resolve())

    return str(candidate)


def resolve_output_dir(output_dir: str) -> Path:
    candidate = Path(output_dir).expanduser()

    if candidate.is_absolute():
        return candidate

    return resolve_repo_path(output_dir)


def build_pipeline(config: AppConfig) -> Dict[str, Any]:
    detector = Detector(
        model_path=resolve_model_path(config.model.model_path),
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
        unsafe_trigger_seconds=config.rules.unsafe_trigger_seconds,
        recent_memory_seconds=config.rules.recent_memory_seconds,
    )

    alert_generator = AlertGenerator()
    visualizer = Visualizer()
    review_engine = IncidentReviewEngine()
    analytics_engine = SiteSafetyAnalyticsEngine()

    return {
        "detector": detector,
        "matcher": matcher,
        "rules_engine": rules_engine,
        "alert_generator": alert_generator,
        "visualizer": visualizer,
        "review_engine": review_engine,
        "analytics_engine": analytics_engine,
    }


# ----------------------------
# Summary helpers
# ----------------------------

def count_states(worker_states: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"safe": 0, "unsafe": 0, "uncertain": 0}
    for worker in worker_states:
        state = str(worker.get("state", "")).lower()
        if state in counts:
            counts[state] += 1
    return counts


def count_alert_levels(alerts: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"info": 0, "warning": 0, "critical": 0}
    for alert in alerts:
        level = str(alert.get("level", "")).lower()
        if level in counts:
            counts[level] += 1
    return counts


def gather_unique_track_ids(frames: List[Dict[str, Any]]) -> Set[int]:
    track_ids: Set[int] = set()
    for frame in frames:
        for worker in frame.get("worker_states", []):
            track_id = worker.get("track_id")
            if track_id is not None:
                try:
                    track_ids.add(int(track_id))
                except (TypeError, ValueError):
                    continue
    return track_ids


def summarize_review_results(review_results: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {
        "likely_compliant": 0,
        "manual_review_required": 0,
        "sustained_likely_violation": 0,
        "insufficient_observation": 0,
    }
    for item in review_results:
        status = str(item.get("review_status", ""))
        if status in summary:
            summary[status] += 1
    return summary


def print_separator() -> None:
    print("-" * 80)


# ----------------------------
# Main processing logic
# ----------------------------

def process_video(
    input_path: Path,
    output_dir: Path,
    pipeline: Dict[str, Any],
    config: AppConfig,
    max_frames: Optional[int] = None,
) -> None:
    detector: Detector = pipeline["detector"]
    matcher: PPEMatcher = pipeline["matcher"]
    rules_engine: SafetyRulesEngine = pipeline["rules_engine"]
    alert_generator: AlertGenerator = pipeline["alert_generator"]
    visualizer: Visualizer = pipeline["visualizer"]
    review_engine: IncidentReviewEngine = pipeline["review_engine"]
    analytics_engine: SiteSafetyAnalyticsEngine = pipeline["analytics_engine"]

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    ensure_dir(output_dir)

    output_video_path = build_output_path(
        output_dir=output_dir,
        input_path=input_path,
        suffix="annotated",
        extension=".mp4",
    )
    output_json_path = build_output_path(
        output_dir=output_dir,
        input_path=input_path,
        suffix="results",
        extension=".json",
    )
    output_review_path = build_output_path(
        output_dir=output_dir,
        input_path=input_path,
        suffix="review",
        extension=".json",
    )
    output_analytics_path = build_output_path(
        output_dir=output_dir,
        input_path=input_path,
        suffix="analytics",
        extension=".json",
    )

    writer = None
    if config.output.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not create output video writer: {output_video_path}")

    all_frame_results: List[Dict[str, Any]] = []
    frame_index = 0
    total_alerts = 0
    final_state_counts = {"safe": 0, "unsafe": 0, "uncertain": 0}
    final_alert_level_counts = {"info": 0, "warning": 0, "critical": 0}

    started_at = datetime.now()

    print_separator()
    print("SITE SAFETY VISION - AUTOMATED VIDEO ANALYSIS")
    print_separator()
    print(f"Input video           : {input_path}")
    print(f"Config file           : loaded successfully")
    print(f"FPS                   : {fps:.2f}")
    print(f"Resolution            : {width} x {height}")
    print(f"Annotated video save  : {'Yes' if config.output.save_video else 'No'}")
    print(f"JSON results save     : {'Yes' if config.output.save_json else 'No'}")
    print_separator()
    print("Processing started...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.track_frame(
                frame=frame,
                tracker_config=config.model.tracker,
                persist=True,
            )

            matched_results = matcher.match(detections)
            worker_states = rules_engine.evaluate_frame(matched_results, fps=fps)
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

            frame_result = {
                "frame_index": frame_index,
                "detections": make_json_safe(detections),
                "matched_results": make_json_safe(matched_results),
                "worker_states": make_json_safe(worker_states),
                "alerts": make_json_safe(alerts),
            }
            all_frame_results.append(frame_result)

            total_alerts += len(alerts)
            final_state_counts = count_states(worker_states)
            final_alert_level_counts = count_alert_levels(alerts)

            if frame_index % 30 == 0:
                print(
                    f"[Frame {frame_index}] "
                    f"workers={len(worker_states)} | "
                    f"safe={final_state_counts['safe']} | "
                    f"unsafe={final_state_counts['unsafe']} | "
                    f"uncertain={final_state_counts['uncertain']} | "
                    f"alerts={len(alerts)}"
                )

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                print(f"Reached max frame limit: {max_frames}")
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    finished_at = datetime.now()
    elapsed = finished_at - started_at

    video_results = {
        "input_file": str(input_path),
        "fps": fps,
        "total_frames_processed": frame_index,
        "frames": all_frame_results,
    }

    if config.output.save_json:
        save_json(make_json_safe(video_results), output_json_path)

    # Auto-run second-level review
    review_results = review_engine.review_video_results(video_results=video_results, fps=fps)
    review_summary = summarize_review_results(review_results)
    save_json({"input_file": str(input_path), "review_results": make_json_safe(review_results)}, output_review_path)

    # Auto-run analytics
    analytics_results = analytics_engine.analyze_video_results(video_results=video_results, fps=fps)
    save_json(make_json_safe(analytics_results), output_analytics_path)

    unique_track_ids = gather_unique_track_ids(all_frame_results)
    overall_analytics = analytics_results.get("overall", {})

    print_separator()
    print("PROCESSING COMPLETED")
    print_separator()
    print(f"Frames processed      : {frame_index}")
    print(f"Unique worker IDs     : {len(unique_track_ids)}")
    print(f"Total alerts generated: {total_alerts}")
    print(f"Elapsed time          : {elapsed}")
    print_separator()
    print("LATEST FRAME STATE SUMMARY")
    print(
        f"Safe: {final_state_counts['safe']} | "
        f"Unsafe: {final_state_counts['unsafe']} | "
        f"Uncertain: {final_state_counts['uncertain']}"
    )
    print(
        f"Info alerts: {final_alert_level_counts['info']} | "
        f"Warning alerts: {final_alert_level_counts['warning']} | "
        f"Critical alerts: {final_alert_level_counts['critical']}"
    )
    print_separator()
    print("SECOND-LEVEL REVIEW SUMMARY")
    print(
        f"Likely compliant: {review_summary['likely_compliant']} | "
        f"Manual review required: {review_summary['manual_review_required']} | "
        f"Sustained likely violation: {review_summary['sustained_likely_violation']} | "
        f"Insufficient observation: {review_summary['insufficient_observation']}"
    )
    print_separator()
    print("ANALYTICS SUMMARY")
    print(f"Average persons detected       : {overall_analytics.get('average_persons_detected', 0):.2f}")
    print(f"Average helmet matched         : {overall_analytics.get('average_helmet_matched', 0):.2f}")
    print(f"Average vest matched           : {overall_analytics.get('average_vest_matched', 0):.2f}")
    print(f"Average full PPE               : {overall_analytics.get('average_full_ppe', 0):.2f}")
    print(f"Helmet compliance rate         : {overall_analytics.get('helmet_compliance_rate', 0):.2%}")
    print(f"Vest compliance rate           : {overall_analytics.get('vest_compliance_rate', 0):.2%}")
    print(f"Full PPE compliance rate       : {overall_analytics.get('full_ppe_compliance_rate', 0):.2%}")
    print_separator()
    print("OUTPUT FILES")
    if config.output.save_video:
        print(f"Annotated video       : {output_video_path}")
    else:
        print("Annotated video       : Not saved (disabled in config)")
    if config.output.save_json:
        print(f"Inference JSON        : {output_json_path}")
    else:
        print("Inference JSON        : Not saved (disabled in config)")
    print(f"Review JSON           : {output_review_path}")
    print(f"Analytics JSON        : {output_analytics_path}")
    print_separator()


# ----------------------------
# Console input flow
# ----------------------------

def ask_video_path() -> Path:
    raw = input("Enter the full path to the input video: ").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("No video path provided.")
    return Path(raw).expanduser()


def ask_config_path() -> Path:
    raw = input(
        "Enter config path [Press Enter for default: configs/inference.yaml]: "
    ).strip().strip('"').strip("'")

    if not raw:
        raw = "configs/inference.yaml"

    return resolve_config_path(raw)


def ask_max_frames() -> Optional[int]:
    raw = input(
        "Enter max frames to process [Press Enter for full video]: "
    ).strip()

    if not raw:
        return None

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("Max frames must be an integer.") from exc

    if value <= 0:
        raise ValueError("Max frames must be greater than 0.")

    return value


def main() -> None:
    try:
        print_separator()
        print("SITE SAFETY VISION")
        print("Single-script automated video analysis")
        print_separator()

        input_path = ask_video_path()
        validate_video_path(input_path)

        config_path = ask_config_path()
        max_frames = ask_max_frames()

        config = load_app_config(config_path)
        output_dir = resolve_output_dir(config.output.output_dir)

        pipeline = build_pipeline(config)

        process_video(
            input_path=input_path,
            output_dir=output_dir,
            pipeline=pipeline,
            config=config,
            max_frames=max_frames,
        )

    except KeyboardInterrupt:
        print("\nProcess cancelled by user.")
        sys.exit(1)
    except Exception as exc:
        print_separator()
        print("ERROR")
        print_separator()
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()