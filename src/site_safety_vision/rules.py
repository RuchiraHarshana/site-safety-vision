from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class WorkerState:
    track_id: int
    state: str
    helmet_seen_recently: bool
    vest_seen_recently: bool
    helmet_missing_frames: int
    vest_missing_frames: int
    uncertain_reasons: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafetyRulesEngine:
    """
    Rule-based safety decision engine with temporal memory.

    Input:
        Per-person matched PPE results from matcher.py

    Output:
        Worker-level state:
        - safe
        - unsafe
        - uncertain

    Core compliance items:
        - helmet
        - vest

    Supplementary items:
        - gloves
        - boots
    """

    VALID_STATES = {"safe", "unsafe", "uncertain"}

    def __init__(
        self,
        required_missing_frames: int = 5,
        recent_memory_frames: int = 15,
    ) -> None:
        """
        Args:
            required_missing_frames:
                Number of consecutive frames a required PPE item can be missing
                before the worker is classified as unsafe, provided visibility
                is reliable.

            recent_memory_frames:
                Number of frames to retain a recently confirmed PPE observation.
                This reduces false unsafe alerts from short-term occlusion.
        """
        self.required_missing_frames = required_missing_frames
        self.recent_memory_frames = recent_memory_frames
        self.worker_memory: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        """Clear all worker memory."""
        self.worker_memory.clear()

    def evaluate_frame(self, matched_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate all matched workers for a single frame.

        Args:
            matched_results:
                Output list from PPEMatcher.match()

        Returns:
            List of worker state dictionaries.
        """
        current_track_ids = set()
        frame_outputs: List[Dict[str, Any]] = []

        for person_result in matched_results:
            track_id = person_result.get("track_id")

            if track_id is None:
                # Skip untracked persons for temporal reasoning.
                # If needed later, a fallback ID strategy can be added.
                continue

            track_id = int(track_id)
            current_track_ids.add(track_id)

            memory = self._get_or_create_worker_memory(track_id)
            self._update_memory_from_match(memory, person_result)
            state = self._decide_state(memory, person_result)

            output = WorkerState(
                track_id=track_id,
                state=state["state"],
                helmet_seen_recently=memory["helmet_seen_recently"],
                vest_seen_recently=memory["vest_seen_recently"],
                helmet_missing_frames=memory["helmet_missing_frames"],
                vest_missing_frames=memory["vest_missing_frames"],
                uncertain_reasons=state["uncertain_reasons"],
                notes=state["notes"],
            )
            frame_outputs.append(output.to_dict())

        self._decay_memory_for_missing_tracks(current_track_ids)
        return frame_outputs

    def _get_or_create_worker_memory(self, track_id: int) -> Dict[str, Any]:
        if track_id not in self.worker_memory:
            self.worker_memory[track_id] = {
                "helmet_seen_recently": False,
                "vest_seen_recently": False,
                "helmet_missing_frames": 0,
                "vest_missing_frames": 0,
                "helmet_recent_counter": 0,
                "vest_recent_counter": 0,
                "frames_since_seen": 0,
            }
        return self.worker_memory[track_id]

    def _update_memory_from_match(self, memory: Dict[str, Any], person_result: Dict[str, Any]) -> None:
        helmet_present = person_result.get("helmet") is not None
        vest_present = person_result.get("vest") is not None

        if helmet_present:
            memory["helmet_seen_recently"] = True
            memory["helmet_recent_counter"] = self.recent_memory_frames
            memory["helmet_missing_frames"] = 0
        else:
            memory["helmet_missing_frames"] += 1

        if vest_present:
            memory["vest_seen_recently"] = True
            memory["vest_recent_counter"] = self.recent_memory_frames
            memory["vest_missing_frames"] = 0
        else:
            memory["vest_missing_frames"] += 1

        if memory["helmet_recent_counter"] > 0:
            memory["helmet_recent_counter"] -= 1
        else:
            memory["helmet_seen_recently"] = False

        if memory["vest_recent_counter"] > 0:
            memory["vest_recent_counter"] -= 1
        else:
            memory["vest_seen_recently"] = False

        memory["frames_since_seen"] = 0

    def _decide_state(self, memory: Dict[str, Any], person_result: Dict[str, Any]) -> Dict[str, Any]:
        visibility = person_result.get("visibility", {})
        notes = list(person_result.get("notes", []))
        uncertain_reasons: List[str] = []

        head_visible = bool(visibility.get("head_region_visible", True))
        torso_visible = bool(visibility.get("torso_region_visible", True))
        person_large_enough = bool(visibility.get("person_large_enough", True))

        helmet_present_now = person_result.get("helmet") is not None
        vest_present_now = person_result.get("vest") is not None

        helmet_seen_recently = bool(memory["helmet_seen_recently"])
        vest_seen_recently = bool(memory["vest_seen_recently"])

        helmet_missing_frames = int(memory["helmet_missing_frames"])
        vest_missing_frames = int(memory["vest_missing_frames"])

        # Small / unclear person -> uncertain
        if not person_large_enough:
            uncertain_reasons.append("Person is too small for reliable PPE verification.")
            return {
                "state": "uncertain",
                "uncertain_reasons": uncertain_reasons,
                "notes": notes,
            }

        # Visibility-limited helmet reasoning
        if not helmet_present_now and not helmet_seen_recently and not head_visible:
            uncertain_reasons.append("Helmet cannot be verified because the head region is unclear.")

        # Visibility-limited vest reasoning
        if not vest_present_now and not vest_seen_recently and not torso_visible:
            uncertain_reasons.append("Vest cannot be verified because the torso region is unclear.")

        if uncertain_reasons:
            return {
                "state": "uncertain",
                "uncertain_reasons": uncertain_reasons,
                "notes": notes,
            }

        # Unsafe: required PPE absent long enough with reliable visibility
        helmet_unverified_for_too_long = (
            not helmet_present_now
            and not helmet_seen_recently
            and head_visible
            and helmet_missing_frames >= self.required_missing_frames
        )

        vest_unverified_for_too_long = (
            not vest_present_now
            and not vest_seen_recently
            and torso_visible
            and vest_missing_frames >= self.required_missing_frames
        )

        if helmet_unverified_for_too_long or vest_unverified_for_too_long:
            if helmet_unverified_for_too_long:
                notes.append("Helmet missing for multiple consecutive frames.")
            if vest_unverified_for_too_long:
                notes.append("Vest missing for multiple consecutive frames.")
            return {
                "state": "unsafe",
                "uncertain_reasons": [],
                "notes": notes,
            }

        # Safe: required PPE is visible now or recently confirmed
        helmet_ok = helmet_present_now or helmet_seen_recently
        vest_ok = vest_present_now or vest_seen_recently

        if helmet_ok and vest_ok:
            return {
                "state": "safe",
                "uncertain_reasons": [],
                "notes": notes,
            }

        # Fallback uncertainty for transitional frames
        if not helmet_ok:
            uncertain_reasons.append("Helmet status is temporarily unresolved.")
        if not vest_ok:
            uncertain_reasons.append("Vest status is temporarily unresolved.")

        return {
            "state": "uncertain",
            "uncertain_reasons": uncertain_reasons,
            "notes": notes,
        }

    def _decay_memory_for_missing_tracks(self, current_track_ids: set[int]) -> None:
        """
        Age memory for workers not seen in the current frame.
        This prevents stale memory from persisting forever.
        """
        tracks_to_delete: List[int] = []

        for track_id, memory in self.worker_memory.items():
            if track_id in current_track_ids:
                continue

            memory["frames_since_seen"] += 1

            if memory["helmet_recent_counter"] > 0:
                memory["helmet_recent_counter"] -= 1
            else:
                memory["helmet_seen_recently"] = False

            if memory["vest_recent_counter"] > 0:
                memory["vest_recent_counter"] -= 1
            else:
                memory["vest_seen_recently"] = False

            # Remove worker memory if absent for too long
            if memory["frames_since_seen"] > self.recent_memory_frames * 2:
                tracks_to_delete.append(track_id)

        for track_id in tracks_to_delete:
            del self.worker_memory[track_id]