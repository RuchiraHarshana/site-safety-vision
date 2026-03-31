from site_safety_vision.rules import SafetyRulesEngine
from site_safety_vision.alerts import AlertGenerator
from site_safety_vision.matcher import PPEMatcher


def test_generate_safe_alert():
    generator = AlertGenerator()

    worker_states = [
        {
            "track_id": 1,
            "state": "safe",
            "helmet_seen_recently": True,
            "vest_seen_recently": True,
            "helmet_missing_frames": 0,
            "vest_missing_frames": 0,
            "uncertain_reasons": [],
            "notes": [],
        }
    ]

    alerts = generator.generate(worker_states)

    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 1
    assert alerts[0]["level"] == "info"
    assert alerts[0]["state"] == "safe"
    assert alerts[0]["message"] == "Worker 1 is compliant."


def test_generate_unsafe_alert_for_helmet():
    generator = AlertGenerator()

    worker_states = [
        {
            "track_id": 2,
            "state": "unsafe",
            "helmet_seen_recently": False,
            "vest_seen_recently": True,
            "helmet_missing_frames": 4,
            "vest_missing_frames": 0,
            "uncertain_reasons": [],
            "notes": ["Helmet missing for multiple consecutive frames."],
        }
    ]

    alerts = generator.generate(worker_states)

    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 2
    assert alerts[0]["level"] == "critical"
    assert alerts[0]["state"] == "unsafe"
    assert "unsafe" in alerts[0]["message"].lower()
    assert "Helmet missing" in alerts[0]["reasons"]


def test_generate_unsafe_alert_for_helmet_and_vest():
    generator = AlertGenerator()

    worker_states = [
        {
            "track_id": 3,
            "state": "unsafe",
            "helmet_seen_recently": False,
            "vest_seen_recently": False,
            "helmet_missing_frames": 5,
            "vest_missing_frames": 5,
            "uncertain_reasons": [],
            "notes": [
                "Helmet missing for multiple consecutive frames.",
                "Vest missing for multiple consecutive frames.",
            ],
        }
    ]

    alerts = generator.generate(worker_states)

    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 3
    assert alerts[0]["level"] == "critical"
    assert alerts[0]["state"] == "unsafe"
    assert "Helmet missing" in alerts[0]["reasons"]
    assert "Vest missing" in alerts[0]["reasons"]
    assert "unsafe" in alerts[0]["message"].lower()


def test_generate_uncertain_alert():
    generator = AlertGenerator()

    worker_states = [
        {
            "track_id": 4,
            "state": "uncertain",
            "helmet_seen_recently": False,
            "vest_seen_recently": False,
            "helmet_missing_frames": 1,
            "vest_missing_frames": 1,
            "uncertain_reasons": [
                "Helmet cannot be verified because the head region is unclear."
            ],
            "notes": ["Head region may be too small or unclear for helmet verification."],
        }
    ]

    alerts = generator.generate(worker_states)

    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 4
    assert alerts[0]["level"] == "warning"
    assert alerts[0]["state"] == "uncertain"
    assert "requires review" in alerts[0]["message"].lower()
    assert len(alerts[0]["reasons"]) == 1


def test_generate_no_alert_for_invalid_worker_state():
    generator = AlertGenerator()

    worker_states = [
        {
            "track_id": None,
            "state": "safe",
            "uncertain_reasons": [],
            "notes": [],
        },
        {
            "track_id": 5,
            "state": "",
            "uncertain_reasons": [],
            "notes": [],
        },
    ]

    alerts = generator.generate(worker_states)

    assert alerts == []