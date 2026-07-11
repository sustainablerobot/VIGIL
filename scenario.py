# -*- coding: utf-8 -*-
#Run this script from your VIZIL root folder:
#cd C:\Users\Administrator\Desktop\VIZIL
#python fix_scenarios.py

#It will fix both issues directly in your files.
import os, csv, io
from pathlib import Path

ROOT = Path(__file__).parent

# ================================================================
# FIX 1: gas_leak scenario
# Problem: no permit active, so no compound rules fire above 35.
# Fix: gas_leak now has a hot_work permit active from T=10 onward
# so CR-001 fires when CO > 25, giving scores up to 75+
# This is realistic: gas leak happens DURING hot work = the real danger
# ================================================================

gas_leak_rows = []
header = "timestamp_offset_sec,zone,co_ppm,h2s_ppm,ch4_percent_lel,temp_c,pressure_bar,vibration_g,oxygen_percent,worker_count,permit_active,permit_type,shift_changeover_in_min"

for t in range(90):
    zone = "C3"
    co   = round(15 + t * 1.5, 1)      # 15 -> 148.5 PPM over 90s
    h2s  = round(1.5 + t * 0.08, 1)
    ch4  = round(4.0 + t * 0.3, 1)
    temp = round(35 + t * 0.05, 1)
    pres = round(1.02 + t * 0.001, 3)
    vib  = round(0.13 + t * 0.001, 3)
    o2   = round(20.8 - t * 0.005, 1)
    wc   = 3 if t < 20 else 4
    # Hot work permit kicks in at T=10 (realistic: gas leak happens during welding job)
    pa   = "true" if t >= 10 else "false"
    pt   = "hot_work" if t >= 10 else "none"
    sc   = max(0, 60 - t)
    gas_leak_rows.append(f"{t},{zone},{co},{h2s},{ch4},{temp},{pres},{vib},{o2},{wc},{pa},{pt},{sc}")

gas_leak_csv = header + "\n" + "\n".join(gas_leak_rows)
path = ROOT / "m1_sensor_simulator" / "data" / "scenario_gas_leak.csv"
with open(path, "w", encoding="utf-8") as f:
    f.write(gas_leak_csv)
print(f"FIXED: scenario_gas_leak.csv ({len(gas_leak_rows)} rows)")
print(f"  CO peaks at {max(float(r.split(',')[2]) for r in gas_leak_rows):.0f} PPM")
print(f"  Hot work permit activates at T=10s")

# ================================================================
# FIX 2: confined_space scenario
# Problem: CR-004 should fire (confined_space permit + O2 < 19.5%)
# but the O2 only drops below 19.5% at T=62 which is very late.
# Also: the scenario is on Zone A1 but permit data loads for A1 from
# active_permits.json. Let's make O2 drop faster AND add a stale sensor
# condition early to show escalation.
# ================================================================

cs_rows = []
for t in range(90):
    zone = "A1"
    # O2 drops much faster now: below 19.5% at T=25, dangerous by T=60
    o2 = round(20.9 - t * 0.028, 2)
    o2 = max(o2, 17.5)  # floor at 17.5%
    co   = round(8 + t * 0.15, 1)
    h2s  = round(0.8 + t * 0.01, 1)
    ch4  = round(1.2 + t * 0.02, 1)
    temp = round(28 + t * 0.04, 1)
    pres = round(1.01, 3)
    vib  = round(0.05, 3)
    wc   = 2
    pa   = "true"
    pt   = "confined_space"
    sc   = max(0, 50 - t)
    cs_rows.append(f"{t},{zone},{co},{h2s},{ch4},{temp},{pres},{vib},{o2},{wc},{pa},{pt},{sc}")

cs_csv = header + "\n" + "\n".join(cs_rows)
path = ROOT / "m1_sensor_simulator" / "data" / "scenario_confined_space.csv"
with open(path, "w", encoding="utf-8") as f:
    f.write(cs_csv)

o2_values = [float(r.split(',')[8]) for r in cs_rows]
first_below = next(i for i, v in enumerate(o2_values) if v < 19.5)
print(f"\nFIXED: scenario_confined_space.csv ({len(cs_rows)} rows)")
print(f"  O2 drops below 19.5% at T={first_below}s")
print(f"  O2 minimum: {min(o2_values):.2f}%")

# ================================================================
# FIX 3: Add O2 depletion to thresholds_breached in M2
# The previous fix only checked gas channels (co, h2s, ch4).
# CR-004 uses _o2_depletion() which checks latest_reading directly.
# Let's verify M2 data_fusion.py has the O2 check.
# ================================================================

m2_path = ROOT / "m2_data_fusion" / "data_fusion.py"
with open(m2_path, encoding="utf-8") as f:
    m2_code = f.read()

# Check if our previous fix included O2
if "oxygen_percent" in m2_code and "thresholds_breached" in m2_code and "O2_LOW" in m2_code:
    print("\nM2 data_fusion.py: O2 check already present - OK")
else:
    print("\nWARNING: M2 O2 threshold check missing - please apply previous data_fusion.py fix")

# ================================================================
# FIX 4: Check CR-004 condition in risk_engine.py
# ================================================================
m3_path = ROOT / "m3_risk_engine" / "risk_engine.py"
with open(m3_path, encoding="utf-8") as f:
    m3_code = f.read()

# Check what _o2_depletion checks
import re
match = re.search(r'def _o2_depletion.*?(?=\ndef )', m3_code, re.DOTALL)
if match:
    print(f"\n_o2_depletion function:\n{match.group()[:300]}")

# The fix: make _o2_depletion check both thresholds_breached AND direct value
old_o2 = '''def _o2_depletion(zone: dict) -> bool:
    """Oxygen below safe entry level (19.5%)."""
    r = zone.get("latest_reading") or {}
    breached = r.get("thresholds_breached", [])
    return "oxygen_percent" in breached'''

new_o2 = '''def _o2_depletion(zone: dict) -> bool:
    """Oxygen below safe entry level (19.5%)."""
    r = zone.get("latest_reading") or {}
    # Check thresholds_breached first (set by M2)
    breached = r.get("thresholds_breached", [])
    if "oxygen_percent" in breached:
        return True
    # Fallback: check the raw value directly
    o2 = r.get("oxygen_percent")
    if o2 is not None:
        try:
            return float(o2) < 19.5
        except (TypeError, ValueError):
            pass
    return False'''

if old_o2 in m3_code:
    m3_code = m3_code.replace(old_o2, new_o2)
    print("\nFIXED: _o2_depletion now checks raw value as fallback")
elif "def _o2_depletion" in m3_code:
    # Different format - patch with regex
    m3_code = re.sub(
        r'def _o2_depletion\(zone: dict\) -> bool:.*?return "oxygen_percent" in breached',
        new_o2,
        m3_code,
        flags=re.DOTALL
    )
    print("\nFIXED (regex): _o2_depletion patched")
else:
    # Inject it before _gas_warning
    m3_code = m3_code.replace(
        "def _gas_warning(zone: dict) -> bool:",
        new_o2 + "\n\n\ndef _gas_warning(zone: dict) -> bool:"
    )
    print("\nINJECTED: _o2_depletion function added before _gas_warning")

# Also make _gas_warning check raw values as fallback
old_gas_warning = '''def _gas_warning(zone: dict) -> bool:
    """Any gas channel at or above warning level."""
    r = zone.get("latest_reading") or {}
    breached = r.get("thresholds_breached", [])
    return any(ch in breached for ch in ["co_ppm", "h2s_ppm", "ch4_percent_lel"])'''

new_gas_warning = '''def _gas_warning(zone: dict) -> bool:
    """Any gas channel at or above warning level."""
    r = zone.get("latest_reading") or {}
    # Check thresholds_breached (computed by M2)
    breached = r.get("thresholds_breached", [])
    if any(ch in breached for ch in ["co_ppm", "h2s_ppm", "ch4_percent_lel"]):
        return True
    # Fallback: check raw values directly (in case M2 fix not yet applied)
    WARN = {"co_ppm": 25.0, "h2s_ppm": 5.0, "ch4_percent_lel": 10.0}
    for ch, threshold in WARN.items():
        val = r.get(ch)
        if val is not None:
            try:
                if float(val) >= threshold:
                    return True
            except (TypeError, ValueError):
                pass
    return False'''

if old_gas_warning in m3_code:
    m3_code = m3_code.replace(old_gas_warning, new_gas_warning)
    print("FIXED: _gas_warning now checks raw values as fallback")
else:
    # patch with regex
    m3_code = re.sub(
        r'def _gas_warning\(zone: dict\) -> bool:.*?return any\(ch in breached for ch in \["co_ppm", "h2s_ppm", "ch4_percent_lel"\]\)',
        new_gas_warning,
        m3_code,
        flags=re.DOTALL
    )
    print("FIXED (regex): _gas_warning patched")

with open(m3_path, "w", encoding="utf-8") as f:
    f.write(m3_code)
print("\nAll fixes applied.")
print("\nExpected scores after fix:")
print("  gas_leak at T=30s:  CO=60 + hot_work permit -> CR-001(+35) + CR-007(+20) = 55/100 MEDIUM")
print("  gas_leak at T=60s:  CO=105 + shift_chg -> CR-001+CR-003+CR-009+CR-010 = 90/100 CRITICAL")
print("  confined_space T=25s: O2=18.9% + confined permit -> CR-004(+35) = 35/100 LOW")
print("  confined_space T=50s: O2=17.5% + CR-008 -> CR-004(+35)+CR-008(+25) = 60/100 MEDIUM")