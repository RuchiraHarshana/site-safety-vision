from site_safety_vision.rules import SafetyRulesEngine
from site_safety_vision.alerts import AlertGenerator
from site_safety_vision.matcher import PPEMatcher


def test_match_helmet_and_vest_to_person():
    matcher = PPEMatcher()

    detections = [
        {
            "class_name": "person",
            "track_id": 1,
            "confidence": 0.95,
            "bbox": [100, 100, 200, 300],
        },
        {
            "class_name": "helmet",
            "track_id": None,
            "confidence": 0.90,
            "bbox": [120, 105, 180, 150],
        },
        {
            "class_name": "vest",
            "track_id": None,
            "confidence": 0.88,
            "bbox": [115, 145, 185, 240],
        },
    ]

    results = matcher.match(detections)

    assert len(results) == 1
    assert results[0]["track_id"] == 1
    assert results[0]["helmet"] is not None
    assert results[0]["vest"] is not None
    assert results[0]["gloves"] == []
    assert results[0]["boots"] == []


def test_match_gloves_and_boots_to_person():
    matcher = PPEMatcher()

    detections = [
        {
            "class_name": "person",
            "track_id": 2,
            "confidence": 0.95,
            "bbox": [100, 100, 200, 320],
        },
        {
            "class_name": "gloves",
            "track_id": None,
            "confidence": 0.80,
            "bbox": [92, 180, 125, 250],
        },
        {
            "class_name": "gloves",
            "track_id": None,
            "confidence": 0.82,
            "bbox": [175, 180, 208, 250],
        },
        {
            "class_name": "boots",
            "track_id": None,
            "confidence": 0.84,
            "bbox": [105, 275, 145, 325],
        },
        {
            "class_name": "boots",
            "track_id": None,
            "confidence": 0.86,
            "bbox": [155, 275, 195, 325],
        },
    ]

    results = matcher.match(detections)

    assert len(results) == 1
    assert results[0]["track_id"] == 2
    assert len(results[0]["gloves"]) >= 1
    assert len(results[0]["boots"]) >= 1


def test_no_ppe_matched_when_items_are_far_from_person():
    matcher = PPEMatcher()

    detections = [
        {
            "class_name": "person",
            "track_id": 3,
            "confidence": 0.94,
            "bbox": [100, 100, 200, 300],
        },
        {
            "class_name": "helmet",
            "track_id": None,
            "confidence": 0.90,
            "bbox": [300, 50, 360, 100],
        },
        {
            "class_name": "vest",
            "track_id": None,
            "confidence": 0.89,
            "bbox": [320, 160, 390, 250],
        },
    ]

    results = matcher.match(detections)

    assert len(results) == 1
    assert results[0]["track_id"] == 3
    assert results[0]["helmet"] is None
    assert results[0]["vest"] is None
    assert results[0]["gloves"] == []
    assert results[0]["boots"] == []


def test_multiple_persons_are_matched_independently():
    matcher = PPEMatcher()

    detections = [
        {
            "class_name": "person",
            "track_id": 10,
            "confidence": 0.96,
            "bbox": [50, 100, 150, 300],
        },
        {
            "class_name": "person",
            "track_id": 11,
            "confidence": 0.97,
            "bbox": [250, 100, 350, 300],
        },
        {
            "class_name": "helmet",
            "track_id": None,
            "confidence": 0.91,
            "bbox": [65, 105, 135, 150],
        },
        {
            "class_name": "vest",
            "track_id": None,
            "confidence": 0.89,
            "bbox": [260, 145, 340, 240],
        },
    ]

    results = matcher.match(detections)

    assert len(results) == 2

    result_by_track = {r["track_id"]: r for r in results}

    assert result_by_track[10]["helmet"] is not None
    assert result_by_track[10]["vest"] is None

    assert result_by_track[11]["helmet"] is None
    assert result_by_track[11]["vest"] is not None


def test_match_returns_empty_when_no_person_detected():
    matcher = PPEMatcher()

    detections = [
        {
            "class_name": "helmet",
            "track_id": None,
            "confidence": 0.92,
            "bbox": [100, 100, 150, 150],
        },
        {
            "class_name": "vest",
            "track_id": None,
            "confidence": 0.90,
            "bbox": [110, 160, 180, 240],
        },
    ]

    results = matcher.match(detections)

    assert results == []