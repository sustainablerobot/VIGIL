# -*- coding: utf-8 -*-
"""
VIGIL — Baseline vs Compound Evaluation
=========================================
Compares a classic single-threshold SCADA alarm (fires the moment any one
sensor channel crosses its "alarm" level) against VIGIL's real compound
risk engine, across every scenario CSV.

This does NOT reimplement either system. It reuses:
  - M1 SensorSimulator's row loading, noise model, and threshold-breach
    detection (m1_sensor_simulator/sensor_simulator.py)
  - M2 DataFusionLayer's real permit/worker/maintenance fusion
    (m2_data_fusion/data_fusion.py)
  - M3 CompoundRiskEngine's real rule-firing loop
    (m3_risk_engine/risk_engine.py) -- the same engine that runs the
    dashboard, evaluated via evaluate_snapshot() exactly as M2 would call it.

The only thing this script adds is the naive baseline comparison and the
timing harness (no MQTT, no real-time sleep -- scenario time comes from
each row's timestamp_offset_sec column).

Run: python evaluate_baseline.py
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "m1_sensor_simulator"))
sys.path.insert(0, str(ROOT_DIR / "m2_data_fusion"))
sys.path.insert(0, str(THIS_DIR))

from sensor_simulator import SensorSimulator  # noqa: E402
from data_fusion import DataFusionLayer  # noqa: E402
from risk_engine import CompoundRiskEngine, SEVERITY_BANDS  # noqa: E402

CRITICAL_SCORE = next(t for t, label in SEVERITY_BANDS if label == "CRITICAL")

SCENARIOS = ["normal", "vizag", "multizone", "confined_space", "gas_leak"]

# Classic single-threshold SCADA baseline: ONE hard alarm setpoint per
# channel, matching common industrial gas-monitor / process-alarm
# conventions (not the multi-tier warning/alarm/emergency ladder M1 uses
# internally -- a real point-alarm panel just has one trip point per
# channel). Oxygen's real-world low-O2 alarm is conventionally 19.5%,
# not the more conservative 18% used for M1's internal "alarm" tier.
NAIVE_ALARM_SETPOINTS = {
    "co_ppm":          {"limit": 50.0, "invert": False},
    "h2s_ppm":         {"limit": 10.0, "invert": False},
    "ch4_percent_lel": {"limit": 20.0, "invert": False},
    "oxygen_percent":  {"limit": 19.5, "invert": True},   # low O2 = danger
    "temp_c":          {"limit": 60.0, "invert": False},
    "vibration_g":     {"limit": 1.0,  "invert": False},
}


def naive_channel_critical(reading) -> bool:
    """
    Fires the instant ANY one sensor channel crosses its single alarm
    setpoint. No compounding, no permit/worker/maintenance context --
    exactly the pattern VIGIL's pitch says is the problem.
    """
    for channel, spec in NAIVE_ALARM_SETPOINTS.items():
        value = getattr(reading, channel, None)
        if value is None:
            continue
        if spec["invert"]:
            if value <= spec["limit"]:
                return True
        else:
            if value >= spec["limit"]:
                return True
    return False


def run_scenario(scenario: str) -> dict:
    """
    Replays one scenario deterministically (no real-time sleep, no noise)
    through the real M1 -> M2 -> M3 pipeline, tick by tick, and records:
      - the first elapsed second the naive baseline would call CRITICAL
      - the first elapsed second VIGIL's real engine scores >= CRITICAL_SCORE
      - whether either side ever fired on this run (for false-positive checks)
    """
    sim = SensorSimulator(
        scenario=scenario,
        add_noise=False,       # deterministic run
        enable_mqtt=False,     # offline
        loop=False,
    )
    fusion = DataFusionLayer(data_dir=ROOT_DIR / "m2_data_fusion" / "data")
    engine = CompoundRiskEngine(min_score_to_alert=1)  # see every score, not just alertable ones

    naive_fired_at = None
    vigil_fired_at = None
    naive_fire_count = 0
    vigil_fire_count = 0

    for tick, row in enumerate(sim._rows):
        elapsed_sec = row.timestamp_offset_sec

        reading = sim._build_reading(row, tick)
        sim._detect_breaches(reading)

        if naive_channel_critical(reading):
            naive_fire_count += 1
            if naive_fired_at is None:
                naive_fired_at = elapsed_sec

        fusion.ingest_sensor_reading(reading)
        snapshot = fusion.fuse_now()

        events = engine.evaluate_snapshot(snapshot)
        for event in events:
            if event.risk_score >= CRITICAL_SCORE:
                vigil_fire_count += 1
                if vigil_fired_at is None:
                    vigil_fired_at = elapsed_sec

    return {
        "scenario": scenario,
        "naive_fired_at": naive_fired_at,
        "vigil_fired_at": vigil_fired_at,
        "naive_fire_count": naive_fire_count,
        "vigil_fire_count": vigil_fire_count,
    }


def format_row(result: dict) -> str:
    scenario = result["scenario"]
    naive = result["naive_fired_at"]
    vigil = result["vigil_fired_at"]

    if naive is None and vigil is None:
        lead = f"neither fired ({result['naive_fire_count']} naive alarms, {result['vigil_fire_count']} VIGIL criticals)"
    elif naive is not None and vigil is not None:
        diff = naive - vigil
        if diff > 0:
            lead = f"+{diff}s earlier"
        elif diff < 0:
            lead = f"{diff}s later"
        else:
            lead = "same tick"
    elif vigil is not None:
        lead = f"baseline never hit alarm level; VIGIL fired at {vigil}s"
    else:
        lead = f"baseline fired at {naive}s; VIGIL never hit CRITICAL (score < {CRITICAL_SCORE}) — worth checking"

    naive_s = "—" if naive is None else str(naive)
    vigil_s = "—" if vigil is None else str(vigil)
    return f"{scenario:<16}{naive_s:>18}{vigil_s:>18}   {lead}"


def main():
    print("\nVIGIL — Naive Single-Threshold Baseline vs Compound Risk Engine")
    print("=" * 78)
    print(f"{'Scenario':<16}{'Naive: 1st crit(s)':>18}{'VIGIL: 1st crit(s)':>18}   Lead time")
    print("-" * 78)

    results = []
    for scenario in SCENARIOS:
        result = run_scenario(scenario)
        results.append(result)
        print(format_row(result))

    print("=" * 78)
    print(
        "\nNaive = first tick any single sensor channel crosses its hard-coded "
        "single alarm setpoint (classic point-threshold SCADA alarm, e.g. "
        "CO>=50ppm, O2<=19.5%).\n"
        f"VIGIL = first tick the real CompoundRiskEngine (same engine as the "
        f"dashboard) scores >= {CRITICAL_SCORE} (CRITICAL) for any zone, run "
        "through the real M1->M2->M3 pipeline (SensorSimulator -> "
        "DataFusionLayer -> CompoundRiskEngine).\n"
        "\nNote on gas_leak: baseline firing while VIGIL stays quiet there is "
        "the INTENDED result, not a bug -- that scenario has no active permit, "
        "so it's a genuine test that VIGIL doesn't over-alert on a rising gas "
        "reading alone. confined_space firing on neither side is the one "
        "worth double-checking (see CR-004's O2 threshold vs this scenario's "
        "actual O2 floor).\n"
    )
    return results


if __name__ == "__main__":
    main()
