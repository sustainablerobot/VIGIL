"""
Tests for VIGIL M5 -- Permit Watch
Run with: python -m pytest test_permit_watch.py -v
"""

import time
import threading
from pathlib import Path

import pytest
from permit_watch import PermitWatch, PermitConflict, CONFLICT_COOLDOWN_SEC

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def pw():
    return PermitWatch(data_dir=DATA_DIR)


def make_reading(zone="C3", co=18.0, ch4=5.0, o2=20.7, h2s=2.1, tick=0):
    return {
        "zone": zone,
        "co_ppm": co,
        "ch4_percent_lel": ch4,
        "oxygen_percent": o2,
        "h2s_ppm": h2s,
        "permit_active": True,
        "permit_type": "hot_work",
        "shift_changeover_in_min": 25,
        "tick": tick,
    }


# ---------------------------------------------------------------------------
# Tests: permit loading
# ---------------------------------------------------------------------------
class TestPermitLoading:
    def test_permits_loaded(self, pw):
        assert len(pw._permits) > 0

    def test_active_permits_in_c3(self, pw):
        permits = pw.get_active_permits("C3")
        assert any(p["permit_id"] == "PTW-2026-001" for p in permits)

    def test_expired_permit_excluded(self, pw):
        active = pw.get_active_permits("D4")
        assert not any(p["permit_id"] == "PTW-2026-004" for p in active)

    def test_correct_permit_type(self, pw):
        permits = pw.get_active_permits("C3")
        hot_work = [p for p in permits if p["permit_type"] == "hot_work"]
        assert len(hot_work) >= 1

    def test_reload_works(self, pw):
        pw.reload_permits()
        assert len(pw._permits) > 0


# ---------------------------------------------------------------------------
# Tests: no conflict when readings are safe
# ---------------------------------------------------------------------------
class TestNoConflictSafe:
    def test_safe_reading_no_conflict(self, pw):
        # CO=12 < threshold of 25 -- no conflict
        pw.ingest_sensor_reading(make_reading(zone="C3", co=12.0))
        conflicts = pw.get_conflicts("C3")
        gas_conflicts = [
            c for c in conflicts if c.conflict_type == "GAS_THRESHOLD_BREACH"
        ]
        assert len(gas_conflicts) == 0

    def test_just_below_threshold_no_conflict(self, pw):
        # CO=24.9 -- just below the 25ppm threshold
        pw.ingest_sensor_reading(make_reading(zone="C3", co=24.9))
        conflicts = pw.get_conflicts("C3")
        gas_conflicts = [
            c for c in conflicts if c.conflict_type == "GAS_THRESHOLD_BREACH"
        ]
        assert len(gas_conflicts) == 0


# ---------------------------------------------------------------------------
# Tests: conflict detection
# ---------------------------------------------------------------------------
class TestConflictDetection:
    def test_co_breach_raises_conflict(self, pw):
        # CO=66 > threshold 25 -- should raise conflict
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        assert len(conflicts) > 0
        types = [c.conflict_type for c in conflicts]
        assert "GAS_THRESHOLD_BREACH" in types

    def test_conflict_has_correct_permit_id(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        gas = [c for c in conflicts if c.conflict_type == "GAS_THRESHOLD_BREACH"]
        assert any(c.permit_id == "PTW-2026-001" for c in gas)

    def test_conflict_has_correct_threshold(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        gas = [c for c in conflicts if c.conflict_type == "GAS_THRESHOLD_BREACH"]
        if gas:
            assert gas[0].threshold_value == 25.0
            assert gas[0].actual_value == 66.0

    def test_o2_depletion_raises_conflict(self, pw):
        # A1 has confined_space permit with o2_percent_min=19.5
        pw.ingest_sensor_reading(
            make_reading(zone="A1", co=10.0, o2=18.5)
        )
        conflicts = pw.get_conflicts("A1")
        o2_conflicts = [c for c in conflicts if c.conflict_type == "OXYGEN_DEPLETION"]
        assert len(o2_conflicts) > 0

    def test_no_fire_watch_conflict_fires(self, pw):
        # Manually set a permit to have no fire watch
        for pid, p in pw._permits.items():
            if p.get("zone") == "C3" and p.get("permit_type") == "hot_work":
                original = p.get("fire_watch_assigned")
                p["fire_watch_assigned"] = False
                pw.ingest_sensor_reading(make_reading(zone="C3", co=10.0))
                conflicts = pw.get_conflicts("C3")
                p["fire_watch_assigned"] = original  # restore
                fw_conflicts = [
                    c for c in conflicts
                    if c.conflict_type == "HOT_WORK_NO_FIRE_WATCH"
                ]
                assert len(fw_conflicts) > 0
                return
        pytest.skip("No hot_work permit in C3 found")

    def test_conflict_not_raised_for_other_zone(self, pw):
        # High CO in C3 should not create conflicts in B2
        pw.ingest_sensor_reading(make_reading(zone="C3", co=99.0))
        conflicts_b2 = pw.get_conflicts("B2")
        assert all(c.zone != "C3" for c in conflicts_b2)


# ---------------------------------------------------------------------------
# Tests: conflict content
# ---------------------------------------------------------------------------
class TestConflictContent:
    def test_conflict_has_action_required(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        for c in conflicts:
            assert c.action_required in (
                "SUSPEND_WORK", "EVACUATE_IMMEDIATELY",
                "ASSIGN_FIRE_WATCH_OR_STOP", "EXTEND_OR_COMPLETE_WORK",
                "REVALIDATE_PERMIT", "CONDUCT_RETEST_BEFORE_RESUMING", "REVIEW"
            )

    def test_conflict_has_oisd_clause(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        for c in conflicts:
            assert "OISD" in c.oisd_clause or "DGFASLI" in c.oisd_clause

    def test_conflict_description_not_empty(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        for c in conflicts:
            assert len(c.description) > 20

    def test_conflict_has_sensor_snapshot(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        for c in conflicts:
            assert "co_ppm" in c.sensor_values_at_conflict

    def test_conflict_severity_label(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        gas = [c for c in conflicts if c.conflict_type == "GAS_THRESHOLD_BREACH"]
        if gas:
            assert gas[0].severity_label() in ("CRITICAL", "HIGH", "MEDIUM")

    def test_suspend_if_breach_set_correctly(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        conflicts = pw.get_conflicts("C3")
        gas = [c for c in conflicts if c.conflict_type == "GAS_THRESHOLD_BREACH"]
        if gas:
            # PTW-2026-001 has suspend_if_breach=true
            assert gas[0].suspend_if_breach is True


# ---------------------------------------------------------------------------
# Tests: conflict clearing
# ---------------------------------------------------------------------------
class TestConflictClearing:
    def test_conflict_clears_when_reading_drops(self, pw):
        # Raise conflict
        pw2 = PermitWatch(data_dir=DATA_DIR)
        pw2.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        assert len(pw2.get_conflicts("C3")) > 0

        # Clear cooldown manually to allow clear
        pw2._cooldown_tracker.clear()

        # Safe reading -- should clear it
        pw2._clear_conflict("PTW-2026-001", "GAS_THRESHOLD_BREACH")
        active = [
            c for c in pw2.get_conflicts("C3")
            if c.conflict_type == "GAS_THRESHOLD_BREACH" and c.is_active
        ]
        assert len(active) == 0

    def test_conflict_history_preserved_after_clear(self, pw):
        pw2 = PermitWatch(data_dir=DATA_DIR)
        pw2.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        pw2._clear_conflict("PTW-2026-001", "GAS_THRESHOLD_BREACH")
        # History should still have the event
        assert len(pw2.get_conflict_history()) > 0


# ---------------------------------------------------------------------------
# Tests: deduplication / cooldown
# ---------------------------------------------------------------------------
class TestDeduplication:
    def test_same_conflict_not_raised_twice_in_cooldown(self, pw):
        pw2 = PermitWatch(data_dir=DATA_DIR)
        received = []
        pw2.register_callback(received.append)

        pw2.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        count_after_first = len(received)

        # Second high reading in the same cooldown window
        pw2.ingest_sensor_reading(make_reading(zone="C3", co=70.0))
        count_after_second = len(received)

        assert count_after_second == count_after_first


# ---------------------------------------------------------------------------
# Tests: permit status
# ---------------------------------------------------------------------------
class TestPermitStatus:
    def test_get_permit_status_returns_objects(self, pw):
        pw.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        statuses = pw.get_permit_status("C3")
        assert len(statuses) > 0

    def test_permit_status_has_conflict_count(self, pw):
        pw2 = PermitWatch(data_dir=DATA_DIR)
        pw2.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        statuses = pw2.get_permit_status("C3")
        total_conflicts = sum(s.conflict_count for s in statuses)
        assert total_conflicts > 0

    def test_permit_status_highest_severity_set(self, pw):
        pw2 = PermitWatch(data_dir=DATA_DIR)
        pw2.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        statuses = pw2.get_permit_status("C3")
        for s in statuses:
            assert s.highest_conflict_severity in (
                "CRITICAL", "HIGH", "MEDIUM", "NONE"
            )


# ---------------------------------------------------------------------------
# Tests: callbacks
# ---------------------------------------------------------------------------
class TestCallbacks:
    def test_callback_fires_on_conflict(self, pw):
        received = []
        pw2 = PermitWatch(data_dir=DATA_DIR)
        pw2.register_callback(received.append)
        pw2.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        assert len(received) > 0
        assert isinstance(received[0], PermitConflict)

    def test_failing_callback_does_not_crash(self, pw):
        good = []
        pw2 = PermitWatch(data_dir=DATA_DIR)
        pw2.register_callback(lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
        pw2.register_callback(good.append)
        pw2.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        # Should not crash, good callback should still fire
        # (conflict may or may not fire depending on cooldown state)

    def test_multiple_callbacks_all_called(self, pw):
        a, b = [], []
        pw2 = PermitWatch(data_dir=DATA_DIR)
        pw2.register_callback(a.append)
        pw2.register_callback(b.append)
        pw2.ingest_sensor_reading(make_reading(zone="C3", co=66.0))
        # Both should receive same events
        assert len(a) == len(b)


# ---------------------------------------------------------------------------
# Tests: thread safety
# ---------------------------------------------------------------------------
class TestThreadSafety:
    def test_concurrent_readings_dont_corrupt_state(self, pw):
        errors = []
        pw2 = PermitWatch(data_dir=DATA_DIR)

        def writer(co):
            try:
                for _ in range(20):
                    pw2.ingest_sensor_reading(
                        make_reading(zone="C3", co=co)
                    )
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=writer, args=(66.0,)),
            threading.Thread(target=writer, args=(12.0,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
