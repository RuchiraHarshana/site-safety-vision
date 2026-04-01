# detector.py 
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


PathLike = Union[str, Path]
ImageInput = Union[str, Path, np.ndarray]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Detector:
    """
    YOLOv8-based detection and tracking wrapper for Site Safety Vision.

    Responsibilities:
    - Load the trained YOLOv8 model once
    - Run image inference
    - Run frame inference
    - Run tracking with ByteTrack
    - Return clean, structured outputs for downstream modules
    """

    def __init__(
        self,
        model_path: PathLike = "models/best.pt",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        image_size: Optional[int] = None,
        tracking_person_confidence_threshold: float = 0.40,
        min_tracking_person_width: float = 40.0,
        min_tracking_person_height: float = 80.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.image_size = image_size

        # Tracking cleanup thresholds for PERSON only
        self.tracking_person_confidence_threshold = tracking_person_confidence_threshold
        self.min_tracking_person_width = min_tracking_person_width
        self.min_tracking_person_height = min_tracking_person_height

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.class_names: Dict[int, str] = self._extract_class_names()

    def _extract_class_names(self) -> Dict[int, str]:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(name) for i, name in enumerate(names)}
        return {}

    def _validate_image_input(self, image: ImageInput) -> Union[str, np.ndarray]:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            return str(image_path)

        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Provided image array is empty.")
            return image

        raise TypeError("image must be a file path or a numpy.ndarray.")

    def _predict_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.image_size is not None:
            kwargs["imgsz"] = self.image_size
        return kwargs

    def _parse_result(self, result: Any, include_tracking: bool = False) -> List[Detection]:
        detections: List[Detection] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
        class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.empty((0,), dtype=int)

        track_ids: List[Optional[int]]
        if include_tracking and getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            track_ids = [None] * len(class_ids)

        for idx in range(len(class_ids)):
            class_id = int(class_ids[idx])
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            bbox = tuple(float(v) for v in xyxy[idx])
            confidence = float(confs[idx])
            track_id = track_ids[idx] if idx < len(track_ids) else None

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    track_id=track_id,
                )
            )

        return detections

    def _is_person_detection(self, detection: Detection) -> bool:
        return detection.class_name.strip().lower() == "person"

    def _is_valid_tracking_person(self, detection: Detection) -> bool:
        if not self._is_person_detection(detection):
            return False

        if detection.confidence < self.tracking_person_confidence_threshold:
            return False

        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1

        if width < self.min_tracking_person_width:
            return False
        if height < self.min_tracking_person_height:
            return False

        return True

    def predict_image(self, image: ImageInput) -> List[Dict[str, Any]]:
        source = self._validate_image_input(image)
        results = self.model.predict(source=source, **self._predict_kwargs())

        if not results:
            return []

        detections = self._parse_result(results[0], include_tracking=False)
        return [det.to_dict() for det in detections]

    def predict_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray.")
        if frame.size == 0:
            return []

        results = self.model.predict(source=frame, **self._predict_kwargs())

        if not results:
            return []

        detections = self._parse_result(results[0], include_tracking=False)
        return [det.to_dict() for det in detections]

    def track_frame(
        self,
        frame: np.ndarray,
        tracker_config: str = "bytetrack.yaml",
        persist: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run tracking on a single frame using YOLOv8 + ByteTrack.

        Important behavior:
        - Return ALL detections so matcher can still see PPE classes
        - Keep track IDs only for stable PERSON detections
        - Remove unstable person track IDs to reduce fragmentation
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray.")
        if frame.size == 0:
            return []

        kwargs = self._predict_kwargs()
        results = self.model.track(
            source=frame,
            tracker=tracker_config,
            persist=persist,
            **kwargs,
        )

        if not results:
            return []

        all_detections = self._parse_result(results[0], include_tracking=True)
        cleaned_detections: List[Dict[str, Any]] = []

        for det in all_detections:
            det_dict = det.to_dict()

            # Keep all PPE detections, but only trust tracking IDs for strong persons
            if self._is_person_detection(det):
                if self._is_valid_tracking_person(det):
                    cleaned_detections.append(det_dict)
                else:
                    # Keep the person box, but remove unstable tracking ID
                    det_dict["track_id"] = None
                    cleaned_detections.append(det_dict)
            else:
                # PPE and other classes remain available to matcher, but should not drive worker tracking
                det_dict["track_id"] = None
                cleaned_detections.append(det_dict)

        return cleaned_detections

    def process_video(
        self,
        video_path: PathLike,
        use_tracking: bool = False,
        tracker_config: str = "bytetrack.yaml",
        persist: bool = True,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        all_results: List[Dict[str, Any]] = []
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if use_tracking:
                    detections = self.track_frame(
                        frame=frame,
                        tracker_config=tracker_config,
                        persist=persist,
                    )
                else:
                    detections = self.predict_frame(frame)

                all_results.append(
                    {
                        "frame_index": frame_index,
                        "detections": detections,
                    }
                )

                frame_index += 1
                if max_frames is not None and frame_index >= max_frames:
                    break
        finally:
            cap.release()

        return all_results

    def iter_video(
        self,
        video_path: PathLike,
        use_tracking: bool = False,
        tracker_config: str = "bytetrack.yaml",
        persist: bool = True,
        max_frames: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if use_tracking:
                    detections = self.track_frame(
                        frame=frame,
                        tracker_config=tracker_config,
                        persist=persist,
                    )
                else:
                    detections = self.predict_frame(frame)

                yield {
                    "frame_index": frame_index,
                    "detections": detections,
                }

                frame_index += 1
                if max_frames is not None and frame_index >= max_frames:
                    break
        finally:
            cap.release()