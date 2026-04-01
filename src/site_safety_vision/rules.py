#rules.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class WorkerState:
    track_id: int
    state: str
    helmet_seen_recently: bool
    vest_seen_recently: bool
    helmet_missing_frames: int
    vest_missing_frames: int
    unsafe_duration: float
    uncertain_duration: float
    risk_level: str
    risk_score: int
    uncertain_reasons: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafetyRulesEngine:
    VALID_STATES = {"safe", "unsafe", "uncertain"}

    def __init__(
        self,
        unsafe_trigger_seconds: float = 5.0,
        recent_memory_seconds: float = 3.0,
        negative_evidence_trigger_seconds: float = 2.0,
    ) -> None:
        self.unsafe_trigger_seconds = unsafe_trigger_seconds
        self.recent_memory_seconds = recent_memory_seconds
        self.negative_evidence_trigger_seconds = negative_evidence_trigger_seconds
        self.worker_memory: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self.worker_memory.clear()

    def evaluate_frame(
        self,
        matched_results: List[Dict[str, Any]],
        fps: float | None = None,
        frame_duration_seconds: float | None = None,
    ) -> List[Dict[str, Any]]:
        current_track_ids = set()
        frame_outputs: List[Dict[str, Any]] = []

        fdur = self._resolve_frame_duration(
            fps=fps,
            frame_duration_seconds=frame_duration_seconds,
        )

        for person_result in matched_results:
            track_id = person_result.get("track_id")
            if track_id is None:
                continue

            track_id = int(track_id)
            current_track_ids.add(track_id)

            memory = self._get_or_create_worker_memory(track_id)

            helmet_present = person_result.get("helmet") is not None
            vest_present = person_result.get("vest") is not None
            no_hardhat_present = person_result.get("no_hardhat") is not None
            no_safety_vest_present = person_result.get("no_safety_vest") is not None

            memory["frames_since_seen"] = 0
            memory["absence_duration_seconds"] = 0.0

            self._decay_recent_memory(memory, fdur)
            self._decay_negative_memory(memory, fdur)

            if helmet_present:
                memory["helmet_missing_frames"] = 0
            else:
                memory["helmet_missing_frames"] += 1

            if vest_present:
                memory["vest_missing_frames"] = 0
            else:
                memory["vest_missing_frames"] += 1

            if helmet_present:
                memory["helmet_missing_duration"] = 0.0
                memory["helmet_seen_recently"] = True
                memory["helmet_recent_duration_remaining"] = self.recent_memory_seconds
                memory["no_hardhat_duration"] = 0.0
                memory["no_hardhat_seen_recently"] = False
                memory["no_hardhat_recent_duration_remaining"] = 0.0
            elif no_hardhat_present:
                memory["no_hardhat_duration"] += fdur
                memory["no_hardhat_seen_recently"] = True
                memory["no_hardhat_recent_duration_remaining"] = self.negative_evidence_trigger_seconds
                memory["helmet_missing_duration"] += fdur
            else:
                memory["helmet_missing_duration"] += fdur

            if vest_present:
                memory["vest_missing_duration"] = 0.0
                memory["vest_seen_recently"] = True
                memory["vest_recent_duration_remaining"] = self.recent_memory_seconds
                memory["no_safety_vest_duration"] = 0.0
                memory["no_safety_vest_seen_recently"] = False
                memory["no_safety_vest_recent_duration_remaining"] = 0.0
            elif no_safety_vest_present:
                memory["no_safety_vest_duration"] += fdur
                memory["no_safety_vest_seen_recently"] = True
                memory["no_safety_vest_recent_duration_remaining"] = self.negative_evidence_trigger_seconds
                memory["vest_missing_duration"] += fdur
            else:
                memory["vest_missing_duration"] += fdur

            state = self._decide_state(memory, person_result)

            # Violation timer update
            if state["state"] == "unsafe":
                memory["unsafe_duration"] += fdur
            else:
                memory["unsafe_duration"] = 0.0

            if state["state"] == "uncertain":
                memory["uncertain_duration"] += fdur
            else:
                memory["uncertain_duration"] = 0.0

            risk_score, risk_level = self._calculate_risk(memory, state["state"])

            output = WorkerState(
                track_id=track_id,
                state=state["state"],
                helmet_seen_recently=memory["helmet_seen_recently"],
                vest_seen_recently=memory["vest_seen_recently"],
                helmet_missing_frames=memory["helmet_missing_frames"],
                vest_missing_frames=memory["vest_missing_frames"],
                unsafe_duration=memory["unsafe_duration"],
                uncertain_duration=memory["uncertain_duration"],
                risk_level=risk_level,
                risk_score=risk_score,
                uncertain_reasons=state["uncertain_reasons"],
                notes=state["notes"],
            )
            frame_outputs.append(output.to_dict())

        self._decay_memory_for_missing_tracks(current_track_ids, fdur)
        return frame_outputs

    def _calculate_risk(self, memory: Dict[str, Any], state: str) -> tuple[int, str]:
        score = 0

        # State-based weighting
        if state == "unsafe":
            score += 3
        elif state == "uncertain":
            score += 1

        # Timer-based weighting
        if memory["unsafe_duration"] >= 2.0:
            score += 2
        elif memory["unsafe_duration"] > 0.0:
            score += 1

        if memory["uncertain_duration"] >= 2.0:
            score += 1

        # Negative evidence weighting
        if memory["no_hardhat_duration"] >= self.negative_evidence_trigger_seconds:
            score += 2
        elif memory["no_hardhat_duration"] > 0.0:
            score += 1

        if memory["no_safety_vest_duration"] >= self.negative_evidence_trigger_seconds:
            score += 1
        elif memory["no_safety_vest_duration"] > 0.0:
            score += 1

        # Long unresolved missing PPE
        if memory["helmet_missing_duration"] >= self.unsafe_trigger_seconds:
            score += 1
        if memory["vest_missing_duration"] >= self.unsafe_trigger_seconds:
            score += 1

        if score >= 4:
            return score, "high"
        if score >= 2:
            return score, "medium"
        return score, "low"

    def _resolve_frame_duration(
        self,
        fps: float | None,
        frame_duration_seconds: float | None,
    ) -> float:
        if frame_duration_seconds is not None and frame_duration_seconds > 0:
            return float(frame_duration_seconds)

        if fps is not None and fps > 0:
            return 1.0 / float(fps)

        return 1.0

    def _get_or_create_worker_memory(self, track_id: int) -> Dict[str, Any]:
        if track_id not in self.worker_memory:
            self.worker_memory[track_id] = {
                "helmet_seen_recently": False,
                "vest_seen_recently": False,
                "helmet_missing_frames": 0,
                "vest_missing_frames": 0,
                "helmet_missing_duration": 0.0,
                "vest_missing_duration": 0.0,
                "helmet_recent_duration_remaining": 0.0,
                "vest_recent_duration_remaining": 0.0,
                "no_hardhat_seen_recently": False,
                "no_safety_vest_seen_recently": False,
                "no_hardhat_duration": 0.0,
                "no_safety_vest_duration": 0.0,
                "no_hardhat_recent_duration_remaining": 0.0,
                "no_safety_vest_recent_duration_remaining": 0.0,
                "frames_since_seen": 0,
                "absence_duration_seconds": 0.0,
                "unsafe_duration": 0.0,
                "uncertain_duration": 0.0,
            }
        return self.worker_memory[track_id]

    def _decay_recent_memory(self, memory: Dict[str, Any], fdur: float) -> None:
        if memory["helmet_recent_duration_remaining"] > 0:
            memory["helmet_recent_duration_remaining"] = max(
                0.0,
                memory["helmet_recent_duration_remaining"] - fdur,
            )
        if memory["helmet_recent_duration_remaining"] <= 0:
            memory["helmet_seen_recently"] = False

        if memory["vest_recent_duration_remaining"] > 0:
            memory["vest_recent_duration_remaining"] = max(
                0.0,
                memory["vest_recent_duration_remaining"] - fdur,
            )
        if memory["vest_recent_duration_remaining"] <= 0:
            memory["vest_seen_recently"] = False

    def _decay_negative_memory(self, memory: Dict[str, Any], fdur: float) -> None:
        if memory["no_hardhat_recent_duration_remaining"] > 0:
            memory["no_hardhat_recent_duration_remaining"] = max(
                0.0,
                memory["no_hardhat_recent_duration_remaining"] - fdur,
            )
        if memory["no_hardhat_recent_duration_remaining"] <= 0:
            memory["no_hardhat_seen_recently"] = False

        if memory["no_safety_vest_recent_duration_remaining"] > 0:
            memory["no_safety_vest_recent_duration_remaining"] = max(
                0.0,
                memory["no_safety_vest_recent_duration_remaining"] - fdur,
            )
        if memory["no_safety_vest_recent_duration_remaining"] <= 0:
            memory["no_safety_vest_seen_recently"] = False

    def _decide_state(
        self,
        memory: Dict[str, Any],
        person_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        visibility = person_result.get("visibility", {})
        notes = list(person_result.get("notes", []))
        uncertain_reasons: List[str] = []

        head_visible = bool(visibility.get("head_region_visible", True))
        torso_visible = bool(visibility.get("torso_region_visible", True))
        person_large_enough = bool(visibility.get("person_large_enough", True))

        helmet_present_now = person_result.get("helmet") is not None
        vest_present_now = person_result.get("vest") is not None
        no_hardhat_present_now = person_result.get("no_hardhat") is not None
        no_safety_vest_present_now = person_result.get("no_safety_vest") is not None

        helmet_seen_recently = bool(memory["helmet_seen_recently"])
        vest_seen_recently = bool(memory["vest_seen_recently"])

        no_hardhat_duration = float(memory["no_hardhat_duration"])
        no_safety_vest_duration = float(memory["no_safety_vest_duration"])
        helmet_missing_duration = float(memory["helmet_missing_duration"])
        vest_missing_duration = float(memory["vest_missing_duration"])

        if not person_large_enough:
            return {
                "state": "uncertain",
                "uncertain_reasons": ["Person too small"],
                "notes": notes,
            }

        helmet_conflict = helmet_present_now and no_hardhat_present_now
        vest_conflict = vest_present_now and no_safety_vest_present_now

        if helmet_conflict:
            uncertain_reasons.append("Conflicting helmet evidence")
        if vest_conflict:
            uncertain_reasons.append("Conflicting vest evidence")

        if uncertain_reasons:
            return {
                "state": "uncertain",
                "uncertain_reasons": uncertain_reasons,
                "notes": notes,
            }

        helmet_negative_strong = (
            no_hardhat_present_now
            and head_visible
            and no_hardhat_duration >= self.negative_evidence_trigger_seconds
        )

        vest_negative_strong = (
            no_safety_vest_present_now
            and torso_visible
            and no_safety_vest_duration >= self.negative_evidence_trigger_seconds
        )

        if helmet_negative_strong or vest_negative_strong:
            if helmet_negative_strong:
                notes.append(f"NO-Hardhat detected ({no_hardhat_duration:.2f}s)")
            if vest_negative_strong:
                notes.append(f"NO-Safety Vest detected ({no_safety_vest_duration:.2f}s)")

            return {
                "state": "unsafe",
                "uncertain_reasons": [],
                "notes": notes,
            }

        helmet_ok = helmet_present_now or helmet_seen_recently
        vest_ok = vest_present_now or vest_seen_recently

        if helmet_ok and vest_ok:
            return {
                "state": "safe",
                "uncertain_reasons": [],
                "notes": notes,
            }

        if not head_visible:
            uncertain_reasons.append("Head not visible")
        if not torso_visible:
            uncertain_reasons.append("Torso not visible")

        if uncertain_reasons:
            return {
                "state": "uncertain",
                "uncertain_reasons": uncertain_reasons,
                "notes": notes,
            }

        helmet_missing_long = (
            not helmet_ok
            and not no_hardhat_present_now
            and helmet_missing_duration >= self.unsafe_trigger_seconds
        )

        vest_missing_long = (
            not vest_ok
            and not no_safety_vest_present_now
            and vest_missing_duration >= self.unsafe_trigger_seconds
        )

        if helmet_missing_long or vest_missing_long:
            if helmet_missing_long:
                notes.append(f"Helmet missing ({helmet_missing_duration:.2f}s)")
            if vest_missing_long:
                notes.append(f"Vest missing ({vest_missing_duration:.2f}s)")

            return {
                "state": "unsafe",
                "uncertain_reasons": [],
                "notes": notes,
            }

        if not helmet_ok:
            uncertain_reasons.append("Helmet unresolved")
        if not vest_ok:
            uncertain_reasons.append("Vest unresolved")

        return {
            "state": "uncertain",
            "uncertain_reasons": uncertain_reasons,
            "notes": notes,
        }

    def _decay_memory_for_missing_tracks(
        self,
        current_track_ids: set[int],
        fdur: float,
    ) -> None:
        tracks_to_delete: List[int] = []

        for track_id, memory in list(self.worker_memory.items()):
            if track_id in current_track_ids:
                continue

            memory["frames_since_seen"] += 1
            memory["absence_duration_seconds"] += fdur

            self._decay_recent_memory(memory, fdur)
            self._decay_negative_memory(memory, fdur)

            if memory["absence_duration_seconds"] > fdur:
                memory["unsafe_duration"] = 0.0
                memory["uncertain_duration"] = 0.0

            cleanup_threshold = max(self.recent_memory_seconds * 2, 1.0)
            epsilon = 1e-9
            if memory["absence_duration_seconds"] + epsilon >= cleanup_threshold:
                tracks_to_delete.append(track_id)

        for track_id in tracks_to_delete:
            del self.worker_memory[track_id]