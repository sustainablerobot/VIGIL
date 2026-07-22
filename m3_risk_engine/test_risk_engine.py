# -*- coding: utf-8 -*-
"""
Tests for VIGIL M3 — Compound Risk Engine
Run with: python -m pytest test_risk_engine.py -v
"""

import time
import threading
from collections import deque
from pathlib import Path

import pytest
from risk_engine import (
    CompoundRiskEngine, RiskEvent, COMPOUND_RULES,
    _gas_warning, _hot_work_active, _confined_space_active,
    _o2_breached, _non_isolated_maintenance, _uncertified_workers_present,
    _changeover_imminent, _stale_sensor, GAS_ESCALATION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_zone(
    zone="C3", co=66.0, h2s=6.2, ch4=16.6, o2=19.9,
    permit_type="hot_work", fire_watch=True,
    isolation_done=False, depressurised=False,
    worker_count=3, has_uncertified=True,
    has_non_isolated=True, changeover=15,
    is_stale=False, breached=None,
):
    breached = breached or ["co_ppm", "h2s_ppm", "ch4_percent_lel"]
    return {
        "zone": zone,
        "latest_reading": {
            "co_ppm": co, "h2s_ppm": h2s, "ch4_percent_lel": ch4,
            "temp_c": 42.0, "pressure_bar": 1.10, "vibration_g": 0.29,
            "oxygen_percent": o2, "worker_count": worker_count,
            "permit_active": permit_type != "none",
            "permit_type": permit_type,
            "shift_changeover_in_min": changeover,
            "scenario": "vizag", "tick": 10,
            "thresholds_breached": breached,
            "breach_levels": {b: "alarm" for b in breached},
            "noise_applied": False,
        },
        "reading_age_sec": 2.0,
        "is_stale": is_stale,
        "active_permits": ([] if permit_type == "none" else [{
            "permit_id": "PTW-001",
            "permit_type": permit_type,
            "issued_to": "Test Worker",
            "work_description": "Test work",
            "gas_test_result_co_ppm": 12,
            "fire_watch_assigned": fire_watch,
        }]),
        "workers_present": [
            {"worker_id": "W-1", "name": "Worker A",
             "role": "welder", "certified_gas_monitor": True},
            {"worker_id": "W-2", "name": "Worker B",
             "role": "operator", "certified_gas_monitor": not has_uncertified},
        ],
        "worker_count": worker_count,
        "maintenance_active": ([] if isolation_done else [{
            "maintenance_id": "MNT-001",
            "equipment_name": "Test Equipment",
            "maintenance_type": "corrective",
            "isolation_done": isolation_done,
            "depressurised": depressurised,
            "notes": "Test maintenance",
        }]),
        "has_uncertified_workers": has_uncertified,
        "has_non_isolated_maintenance": not isolation_done,
        "shift_changeover_in_min": changeover,
    }


def make_global_flags(changeover=True, multi_zone=True, total_workers=6):
    return {
        "shift_changeover_imminent": changeover,
        "multi_zone_permit_active": multi_zone,
        "total_workers_on_site": total_workers,
        "zones_with_active_permits": ["C3", "A1"],
        "zones_with_stale_sensors": [],
        "zones_with_active_maintenance": ["C3"],
    }


@pytest.fixture
def engine():
    return CompoundRiskEngine(
        claude_api_key=None,   # No API key — uses fallback
        cooldown_sec=1,        # Short cooldown for tests
        min_score_to_alert=1,
    )


# ---------------------------------------------------------------------------
# Tests: individual rule condition functions
# ---------------------------------------------------------------------------
class TestRuleConditions:
    def test_gas_warning_true_when_co_breached(self):
        zone = make_zone(breached=["co_ppm"])
        assert _gas_warning(zone) is True

    def test_gas_warning_false_when_no_breach(self):
        zone = make_zone(breached=[], co=5.0, h2s=1.0, ch4=2.0)
        assert _gas_warning(zone) is False

    def test_hot_work_active_true(self):
        zone = make_zone(permit_type="hot_work")
        assert _hot_work_active(zone) is True

    def test_hot_work_active_false_for_confined_space(self):
        zone = make_zone(permit_type="confined_space")
        assert _hot_work_active(zone) is False

    def test_confined_space_active(self):
        zone = make_zone(permit_type="confined_space")
        assert _confined_space_active(zone) is True

    def test_o2_breached_true(self):
        zone = make_zone(breached=["oxygen_percent"])
        assert _o2_breached(zone) is True

    def test_o2_breached_false(self):
        zone = make_zone(breached=["co_ppm"])
        assert _o2_breached(zone) is False

    def test_non_isolated_maintenance_true(self):
        zone = make_zone(isolation_done=False)
        assert _non_isolated_maintenance(zone) is True

    def test_non_isolated_maintenance_false(self):
        zone = make_zone(isolation_done=True)
        assert _non_isolated_maintenance(zone) is False

    def test_uncertified_workers_present(self):
        zone = make_zone(has_uncertified=True, worker_count=2)
        assert _uncertified_workers_present(zone) is True

    def test_uncertified_workers_absent(self):
        zone = make_zone(has_uncertified=False, worker_count=2)
        assert _uncertified_workers_present(zone) is False

    def test_changeover_imminent_true(self):
        zone = make_zone(changeover=25)
        assert _changeover_imminent(zone) is True

    def test_changeover_imminent_false(self):
        zone = make_zone(changeover=90)
        assert _changeover_imminent(zone) is False

    def test_stale_sensor_true(self):
        zone = make_zone(is_stale=True)
        assert _stale_sensor(zone) is True


# ---------------------------------------------------------------------------
# Tests: compound rules fire correctly
# ---------------------------------------------------------------------------
class TestCompoundRules:
    def test_cr001_fires_on_gas_plus_hot_work(self, engine):
        zone = make_zone(breached=["co_ppm"], permit_type="hot_work")
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-001" in rule_ids

    def test_cr001_does_not_fire_without_permit(self, engine):
        zone = make_zone(breached=["co_ppm"], permit_type="none")
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-001" not in rule_ids

    def test_cr002_fires_on_gas_plus_non_isolated_maintenance(self, engine):
        zone = make_zone(breached=["co_ppm"], isolation_done=False)
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-002" in rule_ids

    def test_cr003_fires_on_hot_work_plus_changeover(self, engine):
        zone = make_zone(permit_type="hot_work", changeover=20)
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-003" in rule_ids

    def test_cr004_fires_on_confined_space_plus_o2_depletion(self, engine):
        zone = make_zone(
            permit_type="confined_space",
            breached=["oxygen_percent"],
        )
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-004" in rule_ids

    def test_cr007_fires_on_rapid_escalation(self, engine):
        zone = make_zone(breached=["co_ppm"])
        # Inject rapidly escalating CO history
        engine._gas_history["C3"] = deque(
            [10, 20, 35, 55, 80, 110, 145, 185], maxlen=8
        )
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-007" in rule_ids

    def test_cr007_does_not_fire_on_slow_escalation(self, engine):
        zone = make_zone(breached=["co_ppm"], co=18.0)
        # Slow, gentle rise — below threshold. The current reading (18.0)
        # continues the same gentle slope so _update_gas_history doesn't
        # inject an artificial spike when it appends it.
        engine._gas_history["C3"] = deque(
            [10, 11, 12, 13, 14, 15, 16, 17], maxlen=8
        )
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-007" not in rule_ids

    def test_cr008_fires_on_stale_sensor_with_active_permit(self, engine):
        zone = make_zone(is_stale=True, permit_type="hot_work")
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-008" in rule_ids

    def test_cr009_fires_without_fire_watch(self, engine):
        zone = make_zone(permit_type="hot_work", fire_watch=False)
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-009" in rule_ids

    def test_cr010_fires_on_triple_compound(self, engine):
        zone = make_zone(
            breached=["co_ppm"],
            permit_type="hot_work",
            isolation_done=False,
        )
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        rule_ids = [r["rule_id"] for r in event.rules_fired]
        assert "CR-010" in rule_ids


# ---------------------------------------------------------------------------
# Tests: scoring and severity
# ---------------------------------------------------------------------------
class TestScoring:
    def test_vizag_pattern_scores_critical(self, engine):
        """Full Vizag pattern should score >=80 (CRITICAL)."""
        zone = make_zone()  # defaults = full Vizag pattern
        for co in [18, 24, 31, 42, 49, 57, 66, 75]:
            engine._gas_history.setdefault("C3", deque(maxlen=8)).append(co)
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        assert event.risk_score >= 80
        assert event.severity == "CRITICAL"

    def test_score_capped_at_100(self, engine):
        zone = make_zone()
        for co in [10, 20, 35, 55, 80, 110, 145, 185]:
            engine._gas_history.setdefault("C3", deque(maxlen=8)).append(co)
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        assert event.risk_score <= 100

    def test_normal_conditions_score_low(self, engine):
        zone = make_zone(
            co=12.0, h2s=1.1, ch4=3.0, o2=20.8,
            permit_type="none", isolation_done=True,
            has_uncertified=False, has_non_isolated=False,
            changeover=90, is_stale=False, breached=[],
        )
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags(
            changeover=False, multi_zone=False
        ))
        # No rules fire under normal conditions, so the engine correctly
        # returns None (score below min_score_to_alert) rather than an event.
        assert event is None or event.risk_score < 20

    def test_severity_bands_correct(self, engine):
        from risk_engine import CompoundRiskEngine
        assert engine._score_to_severity(85) == "CRITICAL"
        assert engine._score_to_severity(65) == "HIGH"
        assert engine._score_to_severity(45) == "MEDIUM"
        assert engine._score_to_severity(25) == "LOW"
        assert engine._score_to_severity(10) == "SAFE"


# ---------------------------------------------------------------------------
# Tests: RiskEvent output structure
# ---------------------------------------------------------------------------
class TestRiskEventStructure:
    def test_event_has_all_required_fields(self, engine):
        zone = make_zone()
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        assert event.event_id.startswith("risk-C3-")
        assert event.zone == "C3"
        assert isinstance(event.risk_score, int)
        assert isinstance(event.severity, str)
        assert isinstance(event.rules_fired, list)
        assert isinstance(event.llm_explanation, str)
        assert isinstance(event.recommended_actions, list)
        assert isinstance(event.oisd_clauses, list)
        assert isinstance(event.counterfactual, str)

    def test_rules_fired_have_evidence(self, engine):
        zone = make_zone()
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        for rule in event.rules_fired:
            assert "rule_id" in rule
            assert "evidence" in rule
            assert "oisd_clauses" in rule
            assert "score_contribution" in rule

    def test_oisd_clauses_populated(self, engine):
        zone = make_zone()
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        assert len(event.oisd_clauses) > 0

    def test_counterfactual_references_root_rule(self, engine):
        zone = make_zone()
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        assert "risk score" in event.counterfactual.lower()

    def test_to_json_valid(self, engine):
        import json
        zone = make_zone()
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        parsed = json.loads(event.to_json())
        assert "risk_score" in parsed
        assert "rules_fired" in parsed

    def test_is_actionable_for_high_score(self, engine):
        zone = make_zone()
        event = engine.evaluate_zone_direct("C3", zone, make_global_flags())
        assert event.is_actionable() is True


# ---------------------------------------------------------------------------
# Tests: alert deduplication
# ---------------------------------------------------------------------------
class TestDeduplication:
    def test_same_condition_suppressed_within_cooldown(self, engine):
        zone = make_zone()
        gf = make_global_flags()
        engine._dedup_lock = threading.Lock()

        event1 = engine.evaluate_zone_direct("C3", zone, gf)
        event2 = engine.evaluate_zone_direct("C3", zone, gf)

        assert not event1.alert_suppressed
        assert event2.alert_suppressed

    def test_different_rules_not_suppressed(self, engine):
        zone_a = make_zone(breached=["co_ppm"], permit_type="hot_work", isolation_done=True)
        zone_b = make_zone(breached=[], permit_type="confined_space",
                           has_non_isolated=False, isolation_done=True)

        event_a = engine.evaluate_zone_direct("C3", zone_a, make_global_flags())
        event_b = engine.evaluate_zone_direct("C3", zone_b, make_global_flags())

        # Different rule sets — event_b should not be suppressed
        rules_a = frozenset(r["rule_id"] for r in event_a.rules_fired)
        rules_b = frozenset(r["rule_id"] for r in event_b.rules_fired)

        if rules_a != rules_b:
            assert not event_b.alert_suppressed

    def test_alert_fires_again_after_cooldown(self):
        engine = CompoundRiskEngine(
            claude_api_key=None,
            cooldown_sec=1,
            min_score_to_alert=1,
        )
        zone = make_zone()
        gf = make_global_flags()

        event1 = engine.evaluate_zone_direct("C3", zone, gf)
        time.sleep(1.1)
        event2 = engine.evaluate_zone_direct("C3", zone, gf)

        assert not event1.alert_suppressed
        assert not event2.alert_suppressed


# ---------------------------------------------------------------------------
# Tests: callbacks
# ---------------------------------------------------------------------------
class TestCallbacks:
    def test_callback_fires_for_actionable_event(self, engine):
        received = []
        engine.register_callback(received.append)

        zone = make_zone()
        snapshot = {
            "snapshot_id": "test-snap",
            "global_flags": make_global_flags(),
            "zones": {"C3": zone},
        }
        engine.evaluate_snapshot(snapshot)
        assert len(received) >= 1
        assert isinstance(received[0], RiskEvent)

    def test_failing_callback_does_not_crash(self, engine):
        good = []
        engine.register_callback(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
        engine.register_callback(good.append)

        zone = make_zone()
        snapshot = {
            "snapshot_id": "test-snap",
            "global_flags": make_global_flags(),
            "zones": {"C3": zone},
        }
        engine.evaluate_snapshot(snapshot)
        assert len(good) >= 1


# ---------------------------------------------------------------------------
# Tests: gas history / escalation
# ---------------------------------------------------------------------------
class TestGasHistory:
    def test_history_updated_on_evaluation(self, engine):
        zone = make_zone(co=55.0)
        engine.evaluate_zone_direct("C3", zone, make_global_flags())
        assert "C3" in engine._gas_history
        assert 55.0 in engine._gas_history["C3"]

    def test_history_capped_at_window_size(self, engine):
        zone = make_zone()
        for _ in range(20):
            engine.evaluate_zone_direct("C3", zone, make_global_flags())
        from risk_engine import GAS_HISTORY_WINDOW
        assert len(engine._gas_history["C3"]) <= GAS_HISTORY_WINDOW
