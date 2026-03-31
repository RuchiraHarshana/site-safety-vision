#detector.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

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
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.image_size = image_size

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

    def predict_image(self, image: ImageInput) -> List[Dict[str, Any]]:
        """
        Run object detection on a single image.

        Args:
            image: Image file path or numpy array.

        Returns:
            A list of structured detections.
        """
        source = self._validate_image_input(image)
        results = self.model.predict(source=source, **self._predict_kwargs())

        if not results:
            return []

        detections = self._parse_result(results[0], include_tracking=False)
        return [det.to_dict() for det in detections]

    def predict_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run object detection on a single video frame.

        Args:
            frame: BGR numpy array read by OpenCV.

        Returns:
            A list of structured detections.
        """
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

        Args:
            frame: BGR numpy array.
            tracker_config: Tracker config file name.
            persist: Keep track state across sequential calls.

        Returns:
            A list of structured tracked detections.
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

        detections = self._parse_result(results[0], include_tracking=True)
        return [det.to_dict() for det in detections]

    def process_video(
        self,
        video_path: PathLike,
        use_tracking: bool = False,
        tracker_config: str = "bytetrack.yaml",
        persist: bool = True,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a video file frame by frame.

        Args:
            video_path: Path to the input video.
            use_tracking: Whether to run ByteTrack-based tracking.
            tracker_config: Tracker config file name.
            persist: Keep track state across frames.
            max_frames: Optional cap for number of frames to process.

        Returns:
            A list of per-frame results with frame index and detections.
        """
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
        """
        Yield per-frame video results one at a time.

        Useful for downstream streaming pipelines.

        Args:
            video_path: Path to the input video.
            use_tracking: Whether to run ByteTrack-based tracking.
            tracker_config: Tracker config file name.
            persist: Keep track state across frames.
            max_frames: Optional cap for number of frames to process.

        Yields:
            Dictionary containing frame_index and detections.
        """
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