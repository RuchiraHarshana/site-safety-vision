from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


Color = Tuple[int, int, int]


class Visualizer:
    """
    Visualization utility for Site Safety Vision.

    Responsibilities:
    - draw detection boxes
    - draw worker state labels
    - draw frame-level alert summaries
    - return annotated frames for saving or display
    """

    def __init__(self) -> None:
        self.default_detection_color: Color = (255, 255, 0)
        self.safe_color: Color = (0, 200, 0)
        self.unsafe_color: Color = (0, 0, 255)
        self.uncertain_color: Color = (0, 165, 255)
        self.text_color: Color = (255, 255, 255)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        matched_results: List[Dict[str, Any]],
        worker_states: List[Dict[str, Any]],
        alerts: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Create an annotated copy of the input frame.

        Args:
            frame: Original BGR OpenCV frame.
            detections: Structured detection results from detector.py
            matched_results: Per-person PPE match results from matcher.py
            worker_states: Rule-engine outputs from rules.py
            alerts: Human-readable alerts from alerts.py

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()

        state_by_track_id = {
            int(worker["track_id"]): worker
            for worker in worker_states
            if worker.get("track_id") is not None
        }

        self._draw_detections(annotated, detections, state_by_track_id)
        self._draw_worker_states(annotated, matched_results, state_by_track_id)
        self._draw_alert_summary(annotated, alerts)

        return annotated

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        state_by_track_id: Dict[int, Dict[str, Any]],
    ) -> None:
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue

            class_name = det.get("class_name", "unknown")
            confidence = float(det.get("confidence", 0.0))
            track_id = det.get("track_id")

            label = f"{class_name} {confidence:.2f}"
            color = self.default_detection_color

            if track_id is not None:
                label = f"ID {track_id} | {label}"

            if class_name == "person" and track_id is not None:
                worker = state_by_track_id.get(int(track_id))
                if worker is not None:
                    color = self._get_state_color(worker.get("state", "uncertain"))

            self._draw_box(frame, bbox, label, color)

    def _draw_worker_states(
        self,
        frame: np.ndarray,
        matched_results: List[Dict[str, Any]],
        state_by_track_id: Dict[int, Dict[str, Any]],
    ) -> None:
        for match in matched_results:
            track_id = match.get("track_id")
            person_bbox = match.get("person_bbox")

            if track_id is None or person_bbox is None:
                continue

            worker = state_by_track_id.get(int(track_id))
            if worker is None:
                continue

            state = str(worker.get("state", "uncertain"))
            notes = list(worker.get("notes", []))
            uncertain_reasons = list(worker.get("uncertain_reasons", []))
            color = self._get_state_color(state)

            x1, y1, _, _ = [int(v) for v in person_bbox]

            state_text = f"Worker {track_id}: {state.upper()}"
            self._draw_text(
                frame,
                state_text,
                (x1, max(40, y1 - 28)),
                color,
                font_scale=0.65,
                thickness=2,
            )

            detail_text = self._build_detail_text(match, notes, uncertain_reasons)
            if detail_text:
                self._draw_text(
                    frame,
                    detail_text[:100],
                    (x1, max(60, y1 - 8)),
                    color,
                    font_scale=0.45,
                    thickness=1,
                )

    def _draw_alert_summary(
        self,
        frame: np.ndarray,
        alerts: List[Dict[str, Any]],
    ) -> None:
        if not alerts:
            return

        y_offset = 30
        max_alerts_to_draw = 5

        for alert in alerts[:max_alerts_to_draw]:
            level = str(alert.get("level", "info")).upper()
            message = str(alert.get("message", ""))
            state = str(alert.get("state", "uncertain")).lower()

            color = self._get_state_color(state)
            summary_line = f"[{level}] {message}"

            self._draw_text(
                frame,
                summary_line[:120],
                (20, y_offset),
                color,
                font_scale=0.55,
                thickness=2,
            )
            y_offset += 25

    def _build_detail_text(
        self,
        match: Dict[str, Any],
        notes: List[str],
        uncertain_reasons: List[str],
    ) -> str:
        parts: List[str] = []

        if match.get("helmet") is not None:
            parts.append("helmet")
        if match.get("vest") is not None:
            parts.append("vest")
        if match.get("gloves"):
            parts.append("gloves")
        if match.get("boots"):
            parts.append("boots")

        if parts:
            base = f"Matched: {', '.join(parts)}"
        else:
            base = "No PPE matched"

        if uncertain_reasons:
            return f"{base} | Review: {uncertain_reasons[0]}"

        if notes:
            return f"{base} | {notes[0]}"

        return base

    def _draw_box(
        self,
        frame: np.ndarray,
        bbox: Any,
        label: str,
        color: Color,
        thickness: int = 2,
    ) -> None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text_y = max(20, y1 - 10)
        self._draw_text(
            frame,
            label,
            (x1, text_y),
            color,
            font_scale=0.55,
            thickness=2,
        )

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        color: Color,
        font_scale: float = 0.5,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _get_state_color(self, state: str) -> Color:
        state = state.lower()
        if state == "safe":
            return self.safe_color
        if state == "unsafe":
            return self.unsafe_color
        return self.uncertain_color