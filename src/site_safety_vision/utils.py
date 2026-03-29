from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


BBox = Tuple[float, float, float, float]


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Args:
        path: Directory path.

    Returns:
        Path object for the created/existing directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_file_path(path: str | Path, must_exist: bool = True) -> Path:
    """
    Validate a file path.

    Args:
        path: File path to validate.
        must_exist: Whether the path must already exist.

    Returns:
        Validated Path object.
    """
    file_path = Path(path)

    if must_exist and not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path


def is_image_file(path: str | Path) -> bool:
    """
    Check whether a file path is a supported image file.
    """
    return Path(path).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }


def is_video_file(path: str | Path) -> bool:
    """
    Check whether a file path is a supported video file.
    """
    return Path(path).suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".mpeg",
        ".mpg",
    }


def save_json(data: Dict[str, Any], output_path: str | Path, indent: int = 2) -> Path:
    """
    Save a dictionary as a JSON file.

    Args:
        data: Data to save.
        output_path: Output JSON file path.
        indent: JSON indentation level.

    Returns:
        Path to the saved JSON file.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    return output_path


def bbox_to_tuple(bbox: Any) -> BBox:
    """
    Convert a bbox-like object into a 4-float tuple.

    Expected format:
        (x1, y1, x2, y2)
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox format: {bbox}")

    x1, y1, x2, y2 = bbox
    return float(x1), float(y1), float(x2), float(y2)


def bbox_width(bbox: BBox) -> float:
    x1, _, x2, _ = bbox
    return max(0.0, x2 - x1)


def bbox_height(bbox: BBox) -> float:
    _, y1, _, y2 = bbox
    return max(0.0, y2 - y1)


def bbox_area(bbox: BBox) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def intersection_area(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    return inter_w * inter_h


def iou(box_a: BBox, box_b: BBox) -> float:
    """
    Compute intersection over union.
    """
    inter = intersection_area(box_a, box_b)
    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def intersection_over_candidate(candidate_bbox: BBox, region_bbox: BBox) -> float:
    """
    Compute the overlap ratio relative to the candidate box area.
    Useful for PPE-to-region matching.
    """
    inter = intersection_area(candidate_bbox, region_bbox)
    candidate_area = bbox_area(candidate_bbox)

    if candidate_area <= 0:
        return 0.0

    return inter / candidate_area


def center_in_region(bbox: BBox, region: BBox) -> bool:
    """
    Check whether the bbox center lies inside the given region.
    """
    center_x, center_y = bbox_center(bbox)
    rx1, ry1, rx2, ry2 = region
    return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2


def clamp_bbox_to_frame(
    bbox: BBox,
    frame_width: int,
    frame_height: int,
) -> BBox:
    """
    Clamp bbox coordinates so they stay within frame bounds.
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0.0, min(float(frame_width - 1), x1))
    y1 = max(0.0, min(float(frame_height - 1), y1))
    x2 = max(0.0, min(float(frame_width - 1), x2))
    y2 = max(0.0, min(float(frame_height - 1), y2))

    return x1, y1, x2, y2


def build_output_path(
    output_dir: str | Path,
    input_path: str | Path,
    suffix: str,
    extension: Optional[str] = None,
) -> Path:
    """
    Build an output file path based on an input file name.

    Example:
        input: sample.mp4
        suffix: annotated
        extension: .mp4
        output: sample_annotated.mp4
    """
    output_dir = ensure_dir(output_dir)
    input_path = Path(input_path)

    ext = extension if extension is not None else input_path.suffix
    if ext and not str(ext).startswith("."):
        ext = f".{ext}"

    return output_dir / f"{input_path.stem}_{suffix}{ext}"


def safe_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to int safely.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float safely.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default