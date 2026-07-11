# -*- coding: utf-8 -*-
"""
Tests for VIGIL M2 — Data Fusion Layer
Run with: python -m pytest test_data_fusion.py -v
"""

import json
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from data_fusion import (
    DataFusionLayer, SensorSnapshot, PermitStore, WorkerStore,
    MaintenanceStore, STALE_THRESHOLD_SEC
)

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_reading(zone="C3", co=45.0, tick=5, changeover=20,
                 permit_active=True, permit_type="hot_work",
                 breached=None, breach_levels=None):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "zone": zone,
        "co_ppm": co,
        "h2s_ppm": 3.0,
        "ch4_percent_lel": 12.0,
        "temp_c": 40.0,
        "pressure_bar": 1.08,
        "vibration_g": 0.22,
        "oxygen_percent": 20.2,
        "worker_count": 4,
        "permit_active": permit_active,
        "permit_type": permit_type,
        "shift_changeover_in_min": changeover,
        "scenario": "vizag",
        "tick": tick,
        "thresholds_breached": breached or [],
        "breach_levels": breach_levels or {},
        "noise_applied": False,
    }


@pytest.fixture
def fusion():
    return DataFusionLayer(
        fusion_interval=9999,  # Don't auto-fire during tests
        data_dir=DATA_DIR,
        stale_threshold_sec=STALE_THRESHOLD_SEC,
    )


# ---------------------------------------------------------------------------
# Tests: sensor ingestion
# ---------------------------------------------------------------------------
class TestSensorIngestion:
    def test_ingest_dict_reading(self, fusion):
        reading = make_reading(zone="C3")
        fusion.ingest_sensor_reading(reading)
        assert "C3" in fusion._sensor_buffer

    def test_ingest_object_with_to_dict(self, fusion):
        """Should accept SensorReading objects from M1 too."""
        class FakeReading:
            def to_dict(self):
                return make_reading(zone="B2")
        fusion.ingest_sensor_reading(FakeReading())
        assert "B2" in fusion._sensor_buffer

    def test_latest_reading_overwrites_old(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3", co=10.0, tick=1))
        fusion.ingest_sensor_reading(make_reading(zone="C3", co=99.0, tick=2))
        assert fusion._sensor_buffer["C3"]["co_ppm"] == 99.0

    def test_multiple_zones_buffered_separately(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3", co=30.0))
        fusion.ingest_sensor_reading(make_reading(zone="A1", co=12.0))
        assert fusion._sensor_buffer["C3"]["co_ppm"] == 30.0
        assert fusion._sensor_buffer["A1"]["co_ppm"] == 12.0

    def test_thread_safe_concurrent_ingestion(self, fusion):
        """Multiple threads writing simultaneously should not corrupt buffer."""
        errors = []
        def write(zone, co):
            try:
                for _ in range(50):
                    fusion.ingest_sensor_reading(make_reading(zone=zone, co=co))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write, args=("C3", 45.0)),
            threading.Thread(target=write, args=("B2", 22.0)),
            threading.Thread(target=write, args=("A1", 10.0)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert "C3" in fusion._sensor_buffer
        assert "B2" in fusion._sensor_buffer


# ---------------------------------------------------------------------------
# Tests: snapshot output format
# ---------------------------------------------------------------------------
class TestSnapshotFormat:
    def test_snapshot_has_required_top_level_fields(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        assert snap.snapshot_id.startswith("snap-")
        assert snap.fused_at
        assert isinstance(snap.zones, dict)
        assert isinstance(snap.global_flags, dict)
        assert isinstance(snap.data_sources, dict)

    def test_snapshot_id_is_unique_across_calls(self, fusion):
        fusion.ingest_sensor_reading(make_reading())
        ids = {fusion.fuse_now().snapshot_id for _ in range(5)}
        # IDs include count — all 5 should be unique
        assert len(ids) == 5

    def test_to_json_is_valid(self, fusion):
        fusion.ingest_sensor_reading(make_reading())
        snap = fusion.fuse_now()
        parsed = json.loads(snap.to_json())
        assert isinstance(parsed, dict)
        assert "zones" in parsed

    def test_zone_snapshot_has_all_keys(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        zone = snap.zones.get("C3", {})
        required = [
            "zone", "latest_reading", "reading_age_sec", "is_stale",
            "active_permits", "workers_present", "worker_count",
            "maintenance_active", "has_uncertified_workers",
            "has_non_isolated_maintenance", "shift_changeover_in_min",
        ]
        for key in required:
            assert key in zone, f"Missing key in zone snapshot: {key}"

    def test_get_zone_helper(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        zone = snap.get_zone("C3")
        assert zone is not None
        assert snap.get_zone("NONEXISTENT") is None


# ---------------------------------------------------------------------------
# Tests: staleness detection
# ---------------------------------------------------------------------------
class TestStaleness:
    def test_fresh_reading_not_stale(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        assert not snap.zones["C3"]["is_stale"]

    def test_old_reading_marked_stale(self, fusion):
        """Manually backdate the sensor timestamp to simulate stale sensor."""
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        # Backdate timestamp by more than stale threshold
        stale_time = datetime.now(timezone.utc) - timedelta(seconds=STALE_THRESHOLD_SEC + 10)
        fusion._sensor_timestamps["C3"] = stale_time
        snap = fusion.fuse_now()
        assert snap.zones["C3"]["is_stale"]

    def test_stale_zone_appears_in_global_flags(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        stale_time = datetime.now(timezone.utc) - timedelta(seconds=STALE_THRESHOLD_SEC + 10)
        fusion._sensor_timestamps["C3"] = stale_time
        snap = fusion.fuse_now()
        assert "C3" in snap.global_flags["zones_with_stale_sensors"]

    def test_reading_age_sec_is_accurate(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        time.sleep(0.5)
        snap = fusion.fuse_now()
        age = snap.zones["C3"]["reading_age_sec"]
        assert 0.4 < age < 2.0, f"Reading age {age}s out of expected range"


# ---------------------------------------------------------------------------
# Tests: permit fusion
# ---------------------------------------------------------------------------
class TestPermitFusion:
    def test_active_permit_appears_in_zone(self, fusion):
        """C3 has PTW-2026-001 (hot_work) active in permits.json"""
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        permits = snap.zones["C3"]["active_permits"]
        permit_ids = [p["permit_id"] for p in permits]
        assert "PTW-2026-001" in permit_ids

    def test_expired_permit_not_included(self, fusion):
        """PTW-2026-004 is expired — should not appear anywhere"""
        fusion.ingest_sensor_reading(make_reading(zone="D4"))
        snap = fusion.fuse_now()
        all_permit_ids = []
        for zone_data in snap.zones.values():
            all_permit_ids.extend(p["permit_id"] for p in zone_data["active_permits"])
        assert "PTW-2026-004" not in all_permit_ids

    def test_permit_in_correct_zone_only(self, fusion):
        """PTW-2026-001 is for C3 — must not appear in A1"""
        fusion.ingest_sensor_reading(make_reading(zone="A1"))
        snap = fusion.fuse_now()
        a1_permit_ids = [p["permit_id"] for p in snap.zones.get("A1", {}).get("active_permits", [])]
        assert "PTW-2026-001" not in a1_permit_ids

    def test_zones_with_active_permits_in_global_flags(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        fusion.ingest_sensor_reading(make_reading(zone="A1"))
        snap = fusion.fuse_now()
        assert "C3" in snap.global_flags["zones_with_active_permits"]


# ---------------------------------------------------------------------------
# Tests: worker fusion
# ---------------------------------------------------------------------------
class TestWorkerFusion:
    def test_workers_in_zone_c3(self, fusion):
        """workers.json has W-101, W-102, W-105, W-106 in C3"""
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        worker_ids = [w["worker_id"] for w in snap.zones["C3"]["workers_present"]]
        assert "W-101" in worker_ids

    def test_worker_count_matches_workers_present(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        zone = snap.zones["C3"]
        assert zone["worker_count"] == len(zone["workers_present"])

    def test_uncertified_worker_flag(self, fusion):
        """W-106 (operator) in C3 has certified_gas_monitor=false"""
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        assert snap.zones["C3"]["has_uncertified_workers"] is True

    def test_total_workers_on_site_in_global_flags(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        assert snap.global_flags["total_workers_on_site"] > 0


# ---------------------------------------------------------------------------
# Tests: maintenance fusion
# ---------------------------------------------------------------------------
class TestMaintenanceFusion:
    def test_active_maintenance_in_zone(self, fusion):
        """MNT-2026-041 (Coke Oven Battery #3) is active in C3"""
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        maint_ids = [m["maintenance_id"] for m in snap.zones["C3"]["maintenance_active"]]
        assert "MNT-2026-041" in maint_ids

    def test_completed_maintenance_excluded(self, fusion):
        """MNT-2026-040 is completed — should not appear"""
        snap = fusion.fuse_now()
        all_maint_ids = []
        for zone_data in snap.zones.values():
            all_maint_ids.extend(m["maintenance_id"] for m in zone_data["maintenance_active"])
        assert "MNT-2026-040" not in all_maint_ids

    def test_non_isolated_maintenance_flag(self, fusion):
        """MNT-2026-041 has isolation_done=false → flag should be True"""
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        snap = fusion.fuse_now()
        assert snap.zones["C3"]["has_non_isolated_maintenance"] is True

    def test_properly_isolated_maintenance_no_flag(self, fusion):
        """MNT-2026-042 in B2 has isolation_done=true"""
        fusion.ingest_sensor_reading(make_reading(zone="B2"))
        snap = fusion.fuse_now()
        assert snap.zones["B2"]["has_non_isolated_maintenance"] is False


# ---------------------------------------------------------------------------
# Tests: global flags
# ---------------------------------------------------------------------------
class TestGlobalFlags:
    def test_shift_changeover_imminent_when_within_30_min(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3", changeover=25))
        snap = fusion.fuse_now()
        assert snap.global_flags["shift_changeover_imminent"] is True

    def test_shift_changeover_not_imminent_when_far(self, fusion):
        fusion.ingest_sensor_reading(make_reading(zone="C3", changeover=90))
        snap = fusion.fuse_now()
        assert snap.global_flags["shift_changeover_imminent"] is False

    def test_multi_zone_permit_when_two_zones_active(self, fusion):
        """C3 and A1 both have active permits in permits.json"""
        fusion.ingest_sensor_reading(make_reading(zone="C3"))
        fusion.ingest_sensor_reading(make_reading(zone="A1"))
        snap = fusion.fuse_now()
        assert snap.global_flags["multi_zone_permit_active"] is True


# ---------------------------------------------------------------------------
# Tests: callbacks
# ---------------------------------------------------------------------------
class TestCallbacks:
    def test_callback_receives_snapshot(self, fusion):
        received = []
        fusion.register_callback(received.append)
        fusion.ingest_sensor_reading(make_reading())
        snap = fusion.fuse_now()
        # fuse_now doesn't fire callbacks — only the timer chain does
        # Test the timer chain fires callback
        fusion._fire_callbacks(snap)
        assert len(received) == 1
        assert isinstance(received[0], SensorSnapshot)

    def test_multiple_callbacks_all_called(self, fusion):
        a, b = [], []
        fusion.register_callback(a.append)
        fusion.register_callback(b.append)
        snap = fusion.fuse_now()
        fusion._fire_callbacks(snap)
        assert len(a) == len(b) == 1

    def test_failing_callback_does_not_crash(self, fusion):
        good = []
        fusion.register_callback(lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        fusion.register_callback(good.append)
        snap = fusion.fuse_now()
        fusion._fire_callbacks(snap)
        assert len(good) == 1


# ---------------------------------------------------------------------------
# Tests: timer chain
# ---------------------------------------------------------------------------
class TestTimerChain:
    def test_start_stop_does_not_raise(self, fusion):
        fusion.start()
        time.sleep(0.1)
        fusion.stop()

    def test_periodic_fusion_fires_callback(self):
        received = []
        fusion = DataFusionLayer(
            fusion_interval=1,   # 1 second for test
            data_dir=DATA_DIR,
            stale_threshold_sec=60,
        )
        fusion.register_callback(received.append)
        fusion.ingest_sensor_reading(make_reading())
        fusion.start()
        time.sleep(2.5)  # Wait for 2 fusion cycles
        fusion.stop()
        assert len(received) >= 2, f"Expected >=2 snapshots, got {len(received)}"
