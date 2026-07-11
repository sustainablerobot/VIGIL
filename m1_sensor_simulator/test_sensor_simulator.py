# -*- coding: utf-8 -*-
"""
Tests for VIGIL M1 — Sensor Simulator
Run with: python -m pytest test_sensor_simulator.py -v
"""

import json
import time
import pytest
from pathlib import Path
from sensor_simulator import SensorSimulator, SensorReading, THRESHOLDS


DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def vizag_sim():
    return SensorSimulator(
        scenario="vizag",
        tick_interval=0.01,   # fast for tests
        loop=False,
        enable_mqtt=False,
        add_noise=False,       # deterministic for assertions
        data_dir=DATA_DIR,
    )

@pytest.fixture
def normal_sim():
    return SensorSimulator(
        scenario="normal",
        tick_interval=0.01,
        loop=False,
        enable_mqtt=False,
        add_noise=False,
        data_dir=DATA_DIR,
    )


# ---------------------------------------------------------------------------
# Tests: scenario loading
# ---------------------------------------------------------------------------
class TestScenarioLoading:
    def test_loads_all_four_scenarios(self):
        for name in ["normal", "vizag", "gas_leak", "confined_space"]:
            sim = SensorSimulator(scenario=name, enable_mqtt=False, add_noise=False, data_dir=DATA_DIR)
            assert len(sim._rows) > 0, f"Scenario '{name}' loaded no rows"

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            SensorSimulator(scenario="explosion", enable_mqtt=False, data_dir=DATA_DIR)

    def test_vizag_has_permit_active_rows(self, vizag_sim):
        permit_rows = [r for r in vizag_sim._rows if r.permit_active]
        assert len(permit_rows) > 0, "Vizag scenario should have hot_work permit rows"

    def test_normal_has_no_permit_rows(self, normal_sim):
        permit_rows = [r for r in normal_sim._rows if r.permit_active]
        assert len(permit_rows) == 0, "Normal scenario should have no active permits"


# ---------------------------------------------------------------------------
# Tests: output format
# ---------------------------------------------------------------------------
class TestOutputFormat:
    def test_reading_has_all_required_fields(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()  # loop=False, runs once through

        assert len(collected) > 0
        r = collected[0]

        required = [
            "timestamp", "zone", "co_ppm", "h2s_ppm", "ch4_percent_lel",
            "temp_c", "pressure_bar", "vibration_g", "oxygen_percent",
            "worker_count", "permit_active", "permit_type",
            "shift_changeover_in_min", "scenario", "tick",
            "thresholds_breached", "breach_levels", "noise_applied",
        ]
        d = r.to_dict()
        for field in required:
            assert field in d, f"Missing field: {field}"

    def test_to_json_is_valid_json(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        payload = collected[0].to_json()
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)

    def test_tick_increments(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        ticks = [r.tick for r in collected]
        assert ticks == list(range(len(ticks))), "Ticks must be sequential integers from 0"

    def test_scenario_name_in_output(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        assert all(r.scenario == "vizag" for r in collected)

    def test_timestamp_is_iso8601(self, vizag_sim):
        from datetime import datetime
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        # Should not raise
        datetime.fromisoformat(collected[0].timestamp.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Tests: noise injection
# ---------------------------------------------------------------------------
class TestNoiseInjection:
    def test_no_noise_returns_exact_csv_values(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        # First row of vizag CSV: co_ppm=18, h2s_ppm=2.1
        r = collected[0]
        assert r.co_ppm == 18.0
        assert r.h2s_ppm == 2.1

    def test_noise_changes_values(self):
        sim = SensorSimulator(
            scenario="normal",
            tick_interval=0.01,
            loop=False,
            enable_mqtt=False,
            add_noise=True,
            data_dir=DATA_DIR,
        )
        collected = []
        sim.register_callback(collected.append)
        sim.start()
        # With noise on, at least some values should differ from base
        base_co = sim._rows[0].co_ppm
        # Run several ticks — statistically some will differ
        noisy_values = [r.co_ppm for r in collected]
        assert not all(v == base_co for v in noisy_values), \
            "Noise enabled but all values identical to base"

    def test_noise_never_produces_negative_gas(self):
        sim = SensorSimulator(
            scenario="normal",   # low base values — most susceptible to negative noise
            tick_interval=0.01,
            loop=False,
            enable_mqtt=False,
            add_noise=True,
            data_dir=DATA_DIR,
        )
        collected = []
        sim.register_callback(collected.append)
        sim.start()
        for r in collected:
            assert r.co_ppm >= 0, f"CO went negative: {r.co_ppm}"
            assert r.h2s_ppm >= 0, f"H2S went negative: {r.h2s_ppm}"
            assert r.ch4_percent_lel >= 0, f"CH4 went negative: {r.ch4_percent_lel}"
            assert r.oxygen_percent >= 0, f"O2 went negative: {r.oxygen_percent}"


# ---------------------------------------------------------------------------
# Tests: threshold breach detection
# ---------------------------------------------------------------------------
class TestThresholdBreaches:
    def test_normal_scenario_produces_no_breaches(self, normal_sim):
        collected = []
        normal_sim.register_callback(collected.append)
        normal_sim.start()
        for r in collected:
            assert r.thresholds_breached == [], \
                f"Normal scenario should not breach: tick={r.tick} breached={r.thresholds_breached}"

    def test_vizag_late_ticks_breach_co_and_ch4(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        # Late ticks (index >= 10) should have CO > 50ppm alarm threshold
        late = [r for r in collected if r.tick >= 10]
        assert any("co_ppm" in r.thresholds_breached for r in late), \
            "Vizag scenario tick>=10 should breach CO alarm"

    def test_breach_levels_populated_when_breach_detected(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        breaching = [r for r in collected if r.thresholds_breached]
        if breaching:
            r = breaching[0]
            for ch in r.thresholds_breached:
                assert ch in r.breach_levels, f"breach_levels missing entry for {ch}"
                assert r.breach_levels[ch] in ("warning", "alarm", "emergency")

    def test_oxygen_invert_logic(self):
        """O2 below 19.5% should register as warning."""
        sim = SensorSimulator(
            scenario="confined_space",
            tick_interval=0.01,
            loop=False,
            enable_mqtt=False,
            add_noise=False,
            data_dir=DATA_DIR,
        )
        collected = []
        sim.register_callback(collected.append)
        sim.start()
        # Late ticks have O2 < 19.5 (warning) and < 18 (alarm)
        late = [r for r in collected if r.tick >= 8]
        assert any("oxygen_percent" in r.thresholds_breached for r in late), \
            "Confined space scenario should breach O2 warning"

    def test_emergency_level_set_for_extreme_values(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        # Last tick of vizag: CO=121ppm > 100 emergency threshold
        last = collected[-1]
        if "co_ppm" in last.breach_levels:
            assert last.breach_levels["co_ppm"] == "emergency", \
                f"Expected emergency at CO={last.co_ppm}ppm"


# ---------------------------------------------------------------------------
# Tests: callbacks
# ---------------------------------------------------------------------------
class TestCallbacks:
    def test_multiple_callbacks_all_called(self, vizag_sim):
        results_a, results_b = [], []
        vizag_sim.register_callback(results_a.append)
        vizag_sim.register_callback(results_b.append)
        vizag_sim.start()
        assert len(results_a) == len(results_b) == len(vizag_sim._rows)

    def test_failing_callback_does_not_crash_simulator(self, vizag_sim):
        def bad_callback(r):
            raise RuntimeError("Intentional test error")

        good_results = []
        vizag_sim.register_callback(bad_callback)
        vizag_sim.register_callback(good_results.append)
        # Should not raise
        vizag_sim.start()
        assert len(good_results) > 0, "Good callback should still receive readings"

    def test_async_start_is_non_blocking(self):
        sim = SensorSimulator(
            scenario="normal",
            tick_interval=0.5,
            loop=False,
            enable_mqtt=False,
            add_noise=False,
            data_dir=DATA_DIR,
        )
        t_start = time.monotonic()
        thread = sim.start_async()
        t_after = time.monotonic()
        # start_async should return almost instantly
        assert t_after - t_start < 0.2, "start_async blocked for too long"
        thread.join(timeout=15)


# ---------------------------------------------------------------------------
# Tests: permit context
# ---------------------------------------------------------------------------
class TestPermitContext:
    def test_vizag_permit_type_is_hot_work(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        permit_readings = [r for r in collected if r.permit_active]
        assert all(r.permit_type == "hot_work" for r in permit_readings)

    def test_confined_space_permit_type(self):
        sim = SensorSimulator(
            scenario="confined_space",
            tick_interval=0.01,
            loop=False,
            enable_mqtt=False,
            add_noise=False,
            data_dir=DATA_DIR,
        )
        collected = []
        sim.register_callback(collected.append)
        sim.start()
        assert all(r.permit_type == "confined_space" for r in collected)

    def test_shift_changeover_decrements(self, vizag_sim):
        collected = []
        vizag_sim.register_callback(collected.append)
        vizag_sim.start()
        changeovers = [r.shift_changeover_in_min for r in collected]
        # Each successive row should have a lower (or equal) changeover time
        for i in range(1, len(changeovers)):
            assert changeovers[i] <= changeovers[i - 1], \
                f"Changeover time went UP at tick {i}: {changeovers[i-1]} -> {changeovers[i]}"
