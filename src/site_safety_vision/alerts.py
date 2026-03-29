from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Alert:
    track_id: int
    level: str
    state: str
    message: str
    reasons: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertGenerator:
    """
    Generate human-readable alerts from worker safety states.

    Expected input:
    - output rows from SafetyRulesEngine.evaluate_frame()

    Output:
    - structured alert dictionaries suitable for logging, UI display,
      reporting, or annotated video overlays
    """

    VALID_LEVELS = {"info", "warning", "critical"}

    def generate(self, worker_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate alerts for all workers in a frame.

        Args:
            worker_states:
                List of worker state dictionaries returned by rules.py

        Returns:
            List of structured alerts.
        """
        alerts: List[Dict[str, Any]] = []

        for worker in worker_states:
            alert = self._build_alert(worker)
            if alert is not None:
                alerts.append(alert.to_dict())

        return alerts

    def _build_alert(self, worker: Dict[str, Any]) -> Optional[Alert]:
        track_id = worker.get("track_id")
        state = str(worker.get("state", "")).lower()
        notes = list(worker.get("notes", []))
        uncertain_reasons = list(worker.get("uncertain_reasons", []))

        if track_id is None or not state:
            return None

        if state == "safe":
            return Alert(
                track_id=int(track_id),
                level="info",
                state="safe",
                message=f"Worker {track_id} is compliant.",
                reasons=[],
                notes=notes,
            )

        if state == "unsafe":
            reasons = self._extract_unsafe_reasons(notes)
            message = self._build_unsafe_message(track_id, reasons)

            return Alert(
                track_id=int(track_id),
                level="critical",
                state="unsafe",
                message=message,
                reasons=reasons,
                notes=notes,
            )

        if state == "uncertain":
            message = self._build_uncertain_message(track_id, uncertain_reasons)
            return Alert(
                track_id=int(track_id),
                level="warning",
                state="uncertain",
                message=message,
                reasons=uncertain_reasons,
                notes=notes,
            )

        return None

    def _extract_unsafe_reasons(self, notes: List[str]) -> List[str]:
        reasons: List[str] = []

        for note in notes:
            note_lower = note.lower()

            if "helmet missing" in note_lower:
                reasons.append("Helmet missing")
            elif "vest missing" in note_lower:
                reasons.append("Vest missing")

        return reasons

    def _build_unsafe_message(self, track_id: int, reasons: List[str]) -> str:
        if not reasons:
            return f"Worker {track_id} is unsafe."

        if len(reasons) == 1:
            return f"Worker {track_id} is unsafe: {reasons[0]}."

        joined = " and ".join(reasons)
        return f"Worker {track_id} is unsafe: {joined}."

    def _build_uncertain_message(self, track_id: int, reasons: List[str]) -> str:
        if not reasons:
            return f"Worker {track_id} requires review."

        if len(reasons) == 1:
            return f"Worker {track_id} requires review: {reasons[0]}"

        joined = "; ".join(reasons)
        return f"Worker {track_id} requires review: {joined}"