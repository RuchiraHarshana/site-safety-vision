from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class PersonMatchResult:
    track_id: Optional[int]
    person_bbox: BBox
    helmet: Optional[Dict[str, Any]]
    vest: Optional[Dict[str, Any]]
    gloves: List[Dict[str, Any]]
    boots: List[Dict[str, Any]]
    visibility: Dict[str, bool]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PPEMatcher:
    """
    Match PPE detections to tracked persons using simple geometric heuristics.

    Core compliance items:
    - helmet
    - vest

    Supplementary monitoring items:
    - gloves
    - boots
    """

    PERSON_CLASS = "person"
    HELMET_CLASS = "helmet"
    VEST_CLASS = "vest"
    GLOVES_CLASS = "gloves"
    BOOTS_CLASS = "boots"

    def __init__(
        self,
        helmet_min_overlap: float = 0.05,
        vest_min_overlap: float = 0.10,
        gloves_min_overlap: float = 0.01,
        boots_min_overlap: float = 0.01,
    ) -> None:
        self.helmet_min_overlap = helmet_min_overlap
        self.vest_min_overlap = vest_min_overlap
        self.gloves_min_overlap = gloves_min_overlap
        self.boots_min_overlap = boots_min_overlap

    def match(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match PPE items to each detected person.

        Args:
            detections: List of structured detections from detector.py

        Returns:
            A list of per-person match results.
        """
        persons = [d for d in detections if d.get("class_name") == self.PERSON_CLASS]
        helmets = [d for d in detections if d.get("class_name") == self.HELMET_CLASS]
        vests = [d for d in detections if d.get("class_name") == self.VEST_CLASS]
        gloves = [d for d in detections if d.get("class_name") == self.GLOVES_CLASS]
        boots = [d for d in detections if d.get("class_name") == self.BOOTS_CLASS]

        results: List[Dict[str, Any]] = []

        for person in persons:
            person_bbox = self._bbox_tuple(person["bbox"])
            head_region = self._get_head_region(person_bbox)
            torso_region = self._get_torso_region(person_bbox)
            left_hand_region, right_hand_region = self._get_hand_regions(person_bbox)
            left_foot_region, right_foot_region = self._get_foot_regions(person_bbox)

            helmet_match = self._match_single_best(
                candidates=helmets,
                target_region=head_region,
                min_overlap=self.helmet_min_overlap,
                require_center_inside=False,
            )

            vest_match = self._match_single_best(
                candidates=vests,
                target_region=torso_region,
                min_overlap=self.vest_min_overlap,
                require_center_inside=True,
            )

            glove_matches = self._match_multiple(
                candidates=gloves,
                target_regions=[left_hand_region, right_hand_region],
                min_overlap=self.gloves_min_overlap,
            )

            boot_matches = self._match_multiple(
                candidates=boots,
                target_regions=[left_foot_region, right_foot_region],
                min_overlap=self.boots_min_overlap,
            )

            visibility, notes = self._estimate_visibility(person_bbox, helmet_match, vest_match)

            result = PersonMatchResult(
                track_id=person.get("track_id"),
                person_bbox=person_bbox,
                helmet=helmet_match,
                vest=vest_match,
                gloves=glove_matches,
                boots=boot_matches,
                visibility=visibility,
                notes=notes,
            )
            results.append(result.to_dict())

        return results

    def _match_single_best(
        self,
        candidates: List[Dict[str, Any]],
        target_region: BBox,
        min_overlap: float,
        require_center_inside: bool = False,
    ) -> Optional[Dict[str, Any]]:
        best_candidate: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for candidate in candidates:
            bbox = self._bbox_tuple(candidate["bbox"])
            overlap_ratio = self._intersection_over_candidate(bbox, target_region)

            if overlap_ratio < min_overlap:
                continue

            if require_center_inside and not self._center_in_region(bbox, target_region):
                continue

            score = overlap_ratio * float(candidate.get("confidence", 0.0))
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _match_multiple(
        self,
        candidates: List[Dict[str, Any]],
        target_regions: List[BBox],
        min_overlap: float,
    ) -> List[Dict[str, Any]]:
        matched: List[Dict[str, Any]] = []
        used_indices = set()

        for region in target_regions:
            best_idx = None
            best_score = -1.0

            for idx, candidate in enumerate(candidates):
                if idx in used_indices:
                    continue

                bbox = self._bbox_tuple(candidate["bbox"])
                overlap_ratio = self._intersection_over_candidate(bbox, region)

                if overlap_ratio < min_overlap:
                    continue

                score = overlap_ratio * float(candidate.get("confidence", 0.0))
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                used_indices.add(best_idx)
                matched.append(candidates[best_idx])

        return matched

    def _estimate_visibility(
        self,
        person_bbox: BBox,
        helmet_match: Optional[Dict[str, Any]],
        vest_match: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, bool], List[str]]:
        """
        Estimate whether helmet and vest should be reliably visible.

        This is heuristic and intentionally conservative to support an
        uncertainty-aware decision pipeline later.
        """
        x1, y1, x2, y2 = person_bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        visibility = {
            "head_region_visible": True,
            "torso_region_visible": True,
            "person_large_enough": True,
        }
        notes: List[str] = []

        if height < 80 or width < 25:
            visibility["person_large_enough"] = False
            notes.append("Person appears too small for reliable PPE verification.")

        if helmet_match is None and height < 100:
            visibility["head_region_visible"] = False
            notes.append("Head region may be too small or unclear for helmet verification.")

        if vest_match is None and height < 140:
            visibility["torso_region_visible"] = False
            notes.append("Torso region may be too small or unclear for vest verification.")

        return visibility, notes

    def _get_head_region(self, person_bbox: BBox) -> BBox:
        x1, y1, x2, y2 = person_bbox
        width = x2 - x1
        height = y2 - y1

        return (
            x1 + 0.15 * width,
            y1,
            x2 - 0.15 * width,
            y1 + 0.28 * height,
        )

    def _get_torso_region(self, person_bbox: BBox) -> BBox:
        x1, y1, x2, y2 = person_bbox
        width = x2 - x1
        height = y2 - y1

        return (
            x1 + 0.10 * width,
            y1 + 0.22 * height,
            x2 - 0.10 * width,
            y1 + 0.68 * height,
        )

    def _get_hand_regions(self, person_bbox: BBox) -> Tuple[BBox, BBox]:
        x1, y1, x2, y2 = person_bbox
        width = x2 - x1
        height = y2 - y1

        left_hand = (
            x1 - 0.08 * width,
            y1 + 0.35 * height,
            x1 + 0.20 * width,
            y1 + 0.75 * height,
        )
        right_hand = (
            x2 - 0.20 * width,
            y1 + 0.35 * height,
            x2 + 0.08 * width,
            y1 + 0.75 * height,
        )
        return left_hand, right_hand

    def _get_foot_regions(self, person_bbox: BBox) -> Tuple[BBox, BBox]:
        x1, y1, x2, y2 = person_bbox
        width = x2 - x1
        height = y2 - y1

        left_foot = (
            x1,
            y1 + 0.78 * height,
            x1 + 0.45 * width,
            y2 + 0.08 * height,
        )
        right_foot = (
            x2 - 0.45 * width,
            y1 + 0.78 * height,
            x2,
            y2 + 0.08 * height,
        )
        return left_foot, right_foot

    def _bbox_tuple(self, bbox: Any) -> BBox:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}")
        x1, y1, x2, y2 = bbox
        return float(x1), float(y1), float(x2), float(y2)

    def _intersection_over_candidate(self, candidate_bbox: BBox, region_bbox: BBox) -> float:
        inter_area = self._intersection_area(candidate_bbox, region_bbox)
        candidate_area = self._area(candidate_bbox)
        if candidate_area <= 0:
            return 0.0
        return inter_area / candidate_area

    def _intersection_area(self, a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)

        return inter_w * inter_h

    def _area(self, bbox: BBox) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _center_in_region(self, bbox: BBox, region: BBox) -> bool:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        rx1, ry1, rx2, ry2 = region
        return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2