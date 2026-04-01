#analytics.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class FrameAnalytics:
    frame_index: int
    persons_detected: int
    helmet_matched: int
    vest_matched: int
    full_ppe: int
    helmet_only: int
    vest_only: int
    no_required_ppe_verified: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WindowAnalytics:
    start_frame: int
    end_frame: int
    total_frames: int
    average_persons_detected: float
    average_helmet_matched: float
    average_vest_matched: float
    average_full_ppe: float
    average_helmet_only: float
    average_vest_only: float
    average_no_required_ppe_verified: float
    helmet_compliance_rate: float
    vest_compliance_rate: float
    full_ppe_compliance_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SiteSafetyAnalyticsEngine:
    """
    Site-level temporal analytics engine.

    Reads frame-level matched PPE results and produces average compliance
    statistics over the full video or over configurable time windows.
    """

    def __init__(
        self,
        minimum_presence_frames: int = 1,
    ) -> None:
        self.minimum_presence_frames = minimum_presence_frames

    def analyze_video_results(
        self,
        video_results: Dict[str, Any],
        fps: Optional[float] = None,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        frames = video_results.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError("Invalid video results: 'frames' must be a list.")

        if fps is None or fps <= 0:
            fps = self._infer_fps(video_results)

        frame_analytics = self._build_frame_analytics(frames)

        overall = self._aggregate_window(frame_analytics)

        result: Dict[str, Any] = {
            "fps": fps,
            "total_frames_processed": len(frame_analytics),
            "overall": overall.to_dict(),
            "frame_analytics": [item.to_dict() for item in frame_analytics],
        }

        if window_seconds is not None and window_seconds > 0:
            result["window_seconds"] = window_seconds
            result["window_analytics"] = [
                item.to_dict()
                for item in self._build_window_analytics(
                    frame_analytics=frame_analytics,
                    fps=fps,
                    window_seconds=window_seconds,
                )
            ]

        return result

    def _build_frame_analytics(self, frames: List[Dict[str, Any]]) -> List[FrameAnalytics]:
        frame_analytics: List[FrameAnalytics] = []

        for frame in frames:
            frame_index = int(frame.get("frame_index", 0))
            matched_results = frame.get("matched_results", [])

            persons_detected = 0
            helmet_matched = 0
            vest_matched = 0
            full_ppe = 0
            helmet_only = 0
            vest_only = 0
            no_required_ppe_verified = 0

            for match in matched_results:
                persons_detected += 1

                has_helmet = match.get("helmet") is not None
                has_vest = match.get("vest") is not None

                if has_helmet:
                    helmet_matched += 1
                if has_vest:
                    vest_matched += 1

                if has_helmet and has_vest:
                    full_ppe += 1
                elif has_helmet and not has_vest:
                    helmet_only += 1
                elif has_vest and not has_helmet:
                    vest_only += 1
                else:
                    no_required_ppe_verified += 1

            frame_analytics.append(
                FrameAnalytics(
                    frame_index=frame_index,
                    persons_detected=persons_detected,
                    helmet_matched=helmet_matched,
                    vest_matched=vest_matched,
                    full_ppe=full_ppe,
                    helmet_only=helmet_only,
                    vest_only=vest_only,
                    no_required_ppe_verified=no_required_ppe_verified,
                )
            )

        return frame_analytics

    def _build_window_analytics(
        self,
        frame_analytics: List[FrameAnalytics],
        fps: float,
        window_seconds: float,
    ) -> List[WindowAnalytics]:
        window_size_frames = max(1, int(round(window_seconds * fps)))
        windows: List[WindowAnalytics] = []

        for start_idx in range(0, len(frame_analytics), window_size_frames):
            end_idx = min(start_idx + window_size_frames, len(frame_analytics))
            chunk = frame_analytics[start_idx:end_idx]
            if not chunk:
                continue

            windows.append(self._aggregate_window(chunk))

        return windows

    def _aggregate_window(self, frame_analytics: List[FrameAnalytics]) -> WindowAnalytics:
        if not frame_analytics:
            return WindowAnalytics(
                start_frame=0,
                end_frame=0,
                total_frames=0,
                average_persons_detected=0.0,
                average_helmet_matched=0.0,
                average_vest_matched=0.0,
                average_full_ppe=0.0,
                average_helmet_only=0.0,
                average_vest_only=0.0,
                average_no_required_ppe_verified=0.0,
                helmet_compliance_rate=0.0,
                vest_compliance_rate=0.0,
                full_ppe_compliance_rate=0.0,
            )

        total_frames = len(frame_analytics)

        sum_persons_detected = sum(item.persons_detected for item in frame_analytics)
        sum_helmet_matched = sum(item.helmet_matched for item in frame_analytics)
        sum_vest_matched = sum(item.vest_matched for item in frame_analytics)
        sum_full_ppe = sum(item.full_ppe for item in frame_analytics)
        sum_helmet_only = sum(item.helmet_only for item in frame_analytics)
        sum_vest_only = sum(item.vest_only for item in frame_analytics)
        sum_no_required_ppe_verified = sum(item.no_required_ppe_verified for item in frame_analytics)

        average_persons_detected = sum_persons_detected / total_frames
        average_helmet_matched = sum_helmet_matched / total_frames
        average_vest_matched = sum_vest_matched / total_frames
        average_full_ppe = sum_full_ppe / total_frames
        average_helmet_only = sum_helmet_only / total_frames
        average_vest_only = sum_vest_only / total_frames
        average_no_required_ppe_verified = sum_no_required_ppe_verified / total_frames

        if average_persons_detected > 0:
            helmet_compliance_rate = average_helmet_matched / average_persons_detected
            vest_compliance_rate = average_vest_matched / average_persons_detected
            full_ppe_compliance_rate = average_full_ppe / average_persons_detected
        else:
            helmet_compliance_rate = 0.0
            vest_compliance_rate = 0.0
            full_ppe_compliance_rate = 0.0

        return WindowAnalytics(
            start_frame=frame_analytics[0].frame_index,
            end_frame=frame_analytics[-1].frame_index,
            total_frames=total_frames,
            average_persons_detected=average_persons_detected,
            average_helmet_matched=average_helmet_matched,
            average_vest_matched=average_vest_matched,
            average_full_ppe=average_full_ppe,
            average_helmet_only=average_helmet_only,
            average_vest_only=average_vest_only,
            average_no_required_ppe_verified=average_no_required_ppe_verified,
            helmet_compliance_rate=helmet_compliance_rate,
            vest_compliance_rate=vest_compliance_rate,
            full_ppe_compliance_rate=full_ppe_compliance_rate,
        )

    def _infer_fps(self, video_results: Dict[str, Any]) -> float:
        total_frames = int(video_results.get("total_frames_processed", 0))
        if total_frames <= 0:
            return 30.0
        return 30.0