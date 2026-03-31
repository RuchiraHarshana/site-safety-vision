from site_safety_vision.rules import SafetyRulesEngine
from site_safety_vision.alerts import AlertGenerator
from site_safety_vision.matcher import PPEMatcher


def test_worker_safe_when_helmet_and_vest_present():
    engine = SafetyRulesEngine(unsafe_trigger_seconds=2.0, recent_memory_seconds=3.0)
    matched_results = [
        {
            "track_id": 1,
            "helmet": {"class_name": "helmet", "confidence": 0.95},
            "vest": {"class_name": "vest", "confidence": 0.92},
            "gloves": [],
            "boots": [],
            "visibility": {
                "head_region_visible": True,
                "torso_region_visible": True,
                "person_large_enough": True,
            },
            "notes": [],
        }
    ]
    outputs = engine.evaluate_frame(matched_results, fps=10)
    assert len(outputs) == 1
    assert outputs[0]["track_id"] == 1
    assert outputs[0]["state"] == "safe"


def test_worker_uncertain_when_person_too_small():
    engine = SafetyRulesEngine(unsafe_trigger_seconds=2.0, recent_memory_seconds=3.0)
    matched_results = [
        {
            "track_id": 2,
            "helmet": None,
            "vest": None,
            "gloves": [],
            "boots": [],
            "visibility": {
                "head_region_visible": False,
                "torso_region_visible": False,
                "person_large_enough": False,
            },
            "notes": ["Person appears too small for reliable PPE verification."],
        }
    ]
    outputs = engine.evaluate_frame(matched_results, fps=10)
    assert len(outputs) == 1
    assert outputs[0]["track_id"] == 2
    assert outputs[0]["state"] == "uncertain"
    assert len(outputs[0]["uncertain_reasons"]) > 0


def test_worker_unsafe_after_required_missing_time():
    engine = SafetyRulesEngine(unsafe_trigger_seconds=0.3, recent_memory_seconds=0.2)
    matched_result = {
        "track_id": 3,
        "helmet": None,
        "vest": None,
        "gloves": [],
        "boots": [],
        "visibility": {
            "head_region_visible": True,
            "torso_region_visible": True,
            "person_large_enough": True,
        },
        "notes": [],
    }
    # Simulate 3 frames at 10 FPS (0.1s per frame)
    outputs_1 = engine.evaluate_frame([matched_result], fps=10)
    outputs_2 = engine.evaluate_frame([matched_result], fps=10)
    outputs_3 = engine.evaluate_frame([matched_result], fps=10)
    assert outputs_1[0]["state"] == "uncertain"
    assert outputs_2[0]["state"] == "uncertain"
    assert outputs_3[0]["state"] == "unsafe"


def test_recent_memory_prevents_immediate_unsafe_state():
    engine = SafetyRulesEngine(unsafe_trigger_seconds=0.3, recent_memory_seconds=0.2)
    frame_1 = [
        {
            "track_id": 4,
            "helmet": {"class_name": "helmet", "confidence": 0.93},
            "vest": {"class_name": "vest", "confidence": 0.90},
            "gloves": [],
            "boots": [],
            "visibility": {
                "head_region_visible": True,
                "torso_region_visible": True,
                "person_large_enough": True,
            },
            "notes": [],
        }
    ]
    frame_2 = [
        {
            "track_id": 4,
            "helmet": None,
            "vest": None,
            "gloves": [],
            "boots": [],
            "visibility": {
                "head_region_visible": True,
                "torso_region_visible": True,
                "person_large_enough": True,
            },
            "notes": [],
        }
    ]
    outputs_1 = engine.evaluate_frame(frame_1, fps=10)
    outputs_2 = engine.evaluate_frame(frame_2, fps=10)
    assert outputs_1[0]["state"] == "safe"
    assert outputs_2[0]["state"] == "safe"


def test_missing_track_is_removed_after_long_absence():
    engine = SafetyRulesEngine(unsafe_trigger_seconds=0.3, recent_memory_seconds=0.2)
    frame = [
        {
            "track_id": 5,
            "helmet": {"class_name": "helmet", "confidence": 0.90},
            "vest": {"class_name": "vest", "confidence": 0.88},
            "gloves": [],
            "boots": [],
            "visibility": {
                "head_region_visible": True,
                "torso_region_visible": True,
                "person_large_enough": True,
            },
            "notes": [],
        }
    ]
    engine.evaluate_frame(frame, fps=10)
    assert 5 in engine.worker_memory
    # Simulate 10 empty frames at 10 FPS (1.0s)
    for _ in range(10):
        engine.evaluate_frame([], fps=10)
    assert 5 not in engine.worker_memory