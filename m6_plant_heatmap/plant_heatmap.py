# -*- coding: utf-8 -*-
"""
VIGIL - M6 Plant Heatmap
=========================
Generates a dynamic SVG plant layout where each zone changes color
based on its current risk score from M3.

HOW IT WORKS (simple terms)
----------------------------
Imagine a factory floor map drawn as a grid of rectangles.
Each rectangle = one zone (A1, B2, C3, D4 etc).
Every time M3 fires a RiskEvent, this module re-draws that zone's
rectangle in a new color: green -> yellow -> orange -> red -> dark red.
Workers are shown as small circles inside their zone.
Active permits are shown as a dashed border on the zone.

The map is a pure SVG string — no external image files needed.
Streamlit renders it with st.image() or st.markdown(unsafe_allow_html=True).

WHAT JUDGES SEE
---------------
During the Vizag scenario:
  t=0    : All zones green
  t=2min : Zone C3 turns yellow (CO starting to rise)
  t=4min : Zone C3 turns orange (compound rules firing)
  t=6min : Zone C3 turns red    (HIGH severity)
  t=8min : Zone C3 turns dark red + pulsing border (CRITICAL)
           Workers shown as red dots inside the zone
           Hot work permit shown as dashed warning border

This is the visual moment judges lean forward.

TECHNOLOGIES
------------
- Pure SVG generation (no external libraries needed)
- Inline CSS animations for the pulsing critical alert effect
- Zone layout defined as a simple dict — easy to customise for any plant
"""

import os
os.environ["PYTHONUTF8"] = "1"

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Zone layout definition
# Each zone has: id, display name, grid position (col, row), size, ATEX class
# This defines a simplified coke oven / refinery plant with 8 zones
# ---------------------------------------------------------------------------
ZONE_LAYOUT = [
    # id     label                  col  row  w    h    atex_zone
    ("A1",  "Storage Tank A1",      0,   0,   190, 155, "Zone 2"),
    ("B2",  "Coke Oven B2",         1,   0,   190, 155, "Zone 1"),
    ("C3",  "Battery C3",           0,   1,   190, 155, "Zone 0"),  # highest hazard
    ("D4",  "Workshop D4",          1,   1,   190, 155, "Zone 2"),
]

# SVG canvas settings
CANVAS_W = 400
CANVAS_H = 390
ZONE_GAP = 10
ZONE_OFFSET_X = 20
ZONE_OFFSET_Y = 60   # space for title

# Risk score -> color mapping (matches dashboard SEVERITY_COLORS)
SCORE_COLORS = {
    "SAFE":     ("#2ECC71", "#1a3a2a"),   # (border, fill)
    "LOW":      ("#F4D03F", "#3a3520"),
    "MEDIUM":   ("#E67E22", "#3a2a15"),
    "HIGH":     ("#E74C3C", "#3a1818"),
    "CRITICAL": ("#8B0000", "#4a0a0a"),
}

def score_to_severity(score: int) -> str:
    if score >= 85: return "CRITICAL"
    if score >= 70: return "HIGH"
    if score >= 45: return "MEDIUM"
    if score >= 20: return "LOW"
    return "SAFE"


@dataclass
class ZoneState:
    """Current state of one zone — fed from M3 RiskEvent + M2 snapshot."""
    zone_id: str
    risk_score: int = 0
    severity: str = "SAFE"
    worker_count: int = 0
    worker_positions: List[tuple] = field(default_factory=list)  # (x,y) relative to zone
    has_hot_work_permit: bool = False
    has_confined_space_permit: bool = False
    co_ppm: Optional[float] = None
    is_stale_sensor: bool = False
    permit_type: str = "none"


def generate_plant_svg(
    zone_states: Dict[str, ZoneState],
    show_workers: bool = True,
    show_permits: bool = True,
    show_atex: bool = True,
    pulse_critical: bool = True,
) -> str:
    """
    Generate the full SVG string for the plant heatmap.
    Call this every time any zone state changes and pass to Streamlit.

    Args:
        zone_states: dict of zone_id -> ZoneState (from M3 + M2)
        show_workers: render worker position dots
        show_permits: render permit warning borders
        show_atex:    show ATEX classification label
        pulse_critical: animate critical zones with a pulsing border

    Returns:
        SVG string ready for st.markdown(unsafe_allow_html=True)
    """
    # Build CSS animations for critical pulsing
    css = ""
    if pulse_critical:
        css = """
        <style>
          @keyframes pulse-border {
            0%   { stroke-opacity: 1.0; stroke-width: 3; }
            50%  { stroke-opacity: 0.3; stroke-width: 6; }
            100% { stroke-opacity: 1.0; stroke-width: 3; }
          }
          .critical-pulse { animation: pulse-border 1.2s ease-in-out infinite; }
          @keyframes worker-pulse {
            0%   { r: 6; opacity: 1.0; }
            50%  { r: 9; opacity: 0.6; }
            100% { r: 6; opacity: 1.0; }
          }
          .worker-danger { animation: worker-pulse 1s ease-in-out infinite; }
        </style>
        """

    svg_parts = [
        f'<svg viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#111; border-radius:12px; width:100%;">',
        # CSS animations inside SVG so Streamlit doesn't strip them
        css.replace("<style>", "<style type=\"text/css\">") if css else "",
        # Title bar
        f'<rect x="0" y="0" width="{CANVAS_W}" height="50" fill="#1a1a1a"/>',
        f'<text x="16" y="20" fill="#fff" font-size="14" font-weight="bold" font-family="monospace">'
        f'VIGIL - Plant Zone Heatmap</text>',
        f'<text x="16" y="38" fill="#888" font-size="11" font-family="monospace">'
        f'Live compound risk overlay - Coke Oven Battery Complex</text>',
        # Legend
        _build_legend(CANVAS_W),
    ]

    # Draw each zone
    for (zone_id, label, col, row, w, h, atex) in ZONE_LAYOUT:
        state = zone_states.get(zone_id, ZoneState(zone_id=zone_id))
        sev = state.severity if state.severity in SCORE_COLORS else score_to_severity(state.risk_score)
        border_color, fill_color = SCORE_COLORS[sev]

        x = ZONE_OFFSET_X + col * (w + ZONE_GAP)
        y = ZONE_OFFSET_Y + row * (h + ZONE_GAP)

        is_critical = sev == "CRITICAL"

        # Zone background rectangle
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="8" ry="8" fill="{fill_color}" '
            f'stroke="{border_color}" stroke-width="{"3" if not is_critical else "3"}" '
            f'class="{"critical-pulse" if is_critical and pulse_critical else ""}"/>'
        )

        # Permit warning border (dashed overlay) — shown when permit active
        if show_permits and (state.has_hot_work_permit or state.has_confined_space_permit):
            permit_color = "#FF6B35" if state.has_hot_work_permit else "#9B59B6"
            svg_parts.append(
                f'<rect x="{x+4}" y="{y+4}" width="{w-8}" height="{h-8}" '
                f'rx="5" ry="5" fill="none" '
                f'stroke="{permit_color}" stroke-width="2" stroke-dasharray="6,4" opacity="0.8"/>'
            )

        # Zone ID label (large)
        svg_parts.append(
            f'<text x="{x+12}" y="{y+26}" fill="#fff" '
            f'font-size="16" font-weight="bold" font-family="monospace">'
            f'Zone {zone_id}</text>'
        )

        # Zone name label
        svg_parts.append(
            f'<text x="{x+12}" y="{y+44}" fill="#aaa" '
            f'font-size="10" font-family="monospace">{label}</text>'
        )

        # Risk score badge
        badge_bg = border_color
        svg_parts.append(
            f'<rect x="{x + w - 58}" y="{y+10}" width="50" height="22" '
            f'rx="11" fill="{badge_bg}" opacity="0.9"/>'
        )
        svg_parts.append(
            f'<text x="{x + w - 33}" y="{y+25}" fill="#fff" '
            f'font-size="11" font-weight="bold" font-family="monospace" '
            f'text-anchor="middle">{state.risk_score}/100</text>'
        )

        # Severity label
        svg_parts.append(
            f'<text x="{x+12}" y="{y+64}" fill="{border_color}" '
            f'font-size="12" font-weight="bold" font-family="monospace">{sev}</text>'
        )

        # CO reading
        if state.co_ppm is not None:
            co_color = "#E74C3C" if state.co_ppm > 50 else "#F4D03F" if state.co_ppm > 25 else "#2ECC71"
            stale_marker = " [STALE]" if state.is_stale_sensor else ""
            svg_parts.append(
                f'<text x="{x+12}" y="{y+82}" fill="{co_color}" '
                f'font-size="11" font-family="monospace">'
                f'CO: {state.co_ppm:.1f} ppm{stale_marker}</text>'
            )
        else:
            svg_parts.append(
                f'<text x="{x+12}" y="{y+82}" fill="#555" '
                f'font-size="11" font-family="monospace">CO: waiting...</text>'
            )

        # Permit indicator row
        if show_permits:
            permit_x = x + 12
            permit_y = y + 100
            if state.has_hot_work_permit:
                svg_parts.append(
                    f'<rect x="{permit_x}" y="{permit_y - 11}" width="68" height="15" '
                    f'rx="4" fill="#FF6B35" opacity="0.85"/>'
                )
                svg_parts.append(
                    f'<text x="{permit_x+4}" y="{permit_y}" fill="#fff" '
                    f'font-size="9" font-family="monospace">HOT WORK PTW</text>'
                )
                permit_x += 74
            if state.has_confined_space_permit:
                svg_parts.append(
                    f'<rect x="{permit_x}" y="{permit_y - 11}" width="72" height="15" '
                    f'rx="4" fill="#9B59B6" opacity="0.85"/>'
                )
                svg_parts.append(
                    f'<text x="{permit_x+4}" y="{permit_y}" fill="#fff" '
                    f'font-size="9" font-family="monospace">CONF SPACE PTW</text>'
                )

        # ATEX zone classification
        if show_atex:
            atex_color = "#E74C3C" if "Zone 0" in atex else "#E67E22" if "Zone 1" in atex else "#888"
            svg_parts.append(
                f'<text x="{x + w - 6}" y="{y + h - 8}" fill="{atex_color}" '
                f'font-size="9" font-family="monospace" text-anchor="end" opacity="0.7">'
                f'ATEX {atex}</text>'
            )

        # Worker dots
        if show_workers and state.worker_count > 0:
            _draw_workers(svg_parts, x, y, w, h, state.worker_count, sev, is_critical and pulse_critical)

    svg_parts.append('</svg>')
    svg_string = '\n'.join(svg_parts)
    # Wrap in full HTML document for st.components.v1.html()
    # This avoids Streamlit stripping <style> tags from st.markdown()
    return f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin:0; padding:0; }}
  html, body {{ 
    background: #0e1117;
    width: 100%; 
    height: 100%; 
    overflow: hidden;
  }}
  svg {{ 
    display:block; 
    width:100%; 
    height:auto;
    border-radius: 12px;
  }}
</style>
</head><body>{svg_string}</body></html>'''


def _draw_workers(svg_parts, zone_x, zone_y, zone_w, zone_h,
                  count, severity, animate):
    """Draw worker position dots inside a zone."""
    # Fixed positions within the zone for up to 8 workers
    positions = [
        (0.25, 0.75), (0.45, 0.75), (0.65, 0.75), (0.80, 0.75),
        (0.25, 0.88), (0.45, 0.88), (0.65, 0.88), (0.80, 0.88),
    ]
    worker_color = "#E74C3C" if severity in ("HIGH", "CRITICAL") else "#3498DB"
    css_class = "worker-danger" if animate else ""

    for i in range(min(count, len(positions))):
        rel_x, rel_y = positions[i]
        wx = zone_x + int(rel_x * zone_w)
        wy = zone_y + int(rel_y * zone_h)
        svg_parts.append(
            f'<circle cx="{wx}" cy="{wy}" r="6" fill="{worker_color}" '
            f'opacity="0.9" class="{css_class}"/>'
        )
        # Hard hat icon (simple triangle on top of circle)
        svg_parts.append(
            f'<polygon points="{wx},{wy-10} {wx-5},{wy-4} {wx+5},{wy-4}" '
            f'fill="{worker_color}" opacity="0.7"/>'
        )

    # Worker count label
    if count > 0:
        svg_parts.append(
            f'<text x="{zone_x + zone_w - 8}" y="{zone_y + zone_h - 8}" '
            f'fill="{worker_color}" font-size="11" font-weight="bold" '
            f'font-family="monospace" text-anchor="end">'
            f'{count} worker{"s" if count > 1 else ""}</text>'
        )


def _build_legend(canvas_w: int) -> str:
    """Build the color legend strip at the bottom of the SVG."""
    legend_y = 360
    legend_items = [
        ("SAFE",     "#2ECC71", "0-19"),
        ("LOW",      "#F4D03F", "20-44"),
        ("MEDIUM",   "#E67E22", "45-69"),
        ("HIGH",     "#E74C3C", "70-84"),
        ("CRITICAL", "#8B0000", "85-100"),
    ]
    parts = [f'<rect x="0" y="{legend_y-4}" width="{canvas_w}" height="34" fill="#1a1a1a"/>']
    x = 14
    for label, color, rng in legend_items:
        parts.append(f'<rect x="{x}" y="{legend_y+2}" width="12" height="12" rx="3" fill="{color}"/>')
        parts.append(
            f'<text x="{x+16}" y="{legend_y+13}" fill="#aaa" '
            f'font-size="9" font-family="monospace">{label} ({rng})</text>'
        )
        x += 78
    return '\n'.join(parts)


def build_zone_states_from_pipeline(pipeline_state: dict) -> Dict[str, ZoneState]:
    """
    Convert the dashboard's pipeline.get_state() dict into ZoneState objects
    for heatmap rendering.

    This is the bridge function called from dashboard.py.
    Input: raw dict from VigilPipeline.get_state()
    Output: dict of zone_id -> ZoneState ready for generate_plant_svg()
    """
    zone_states = {}

    for zone_id in [z[0] for z in ZONE_LAYOUT]:
        risk_event = pipeline_state.get("risk_events", {}).get(zone_id, {})
        sensor_data = pipeline_state.get("zone_states", {}).get(zone_id, {})

        # Determine permit types
        permits = sensor_data.get("active_permits", [])
        permit_type = sensor_data.get("permit_type", "none")
        has_hot_work = "hot_work" in permit_type or any(
            "hot" in str(p).lower() for p in permits
        )
        has_confined = "confined" in permit_type or any(
            "confined" in str(p).lower() for p in permits
        )

        # Stale sensor: CR-008 fired
        rules_fired = risk_event.get("rules_fired", [])
        is_stale = any(r.get("rule_id") == "CR-008" for r in rules_fired)

        # Worker count from sensor data
        worker_count = sensor_data.get("worker_count", 0)

        zone_states[zone_id] = ZoneState(
            zone_id=zone_id,
            risk_score=risk_event.get("risk_score", 0),
            severity=risk_event.get("severity", "SAFE"),
            worker_count=worker_count,
            has_hot_work_permit=has_hot_work,
            has_confined_space_permit=has_confined,
            co_ppm=sensor_data.get("co_ppm"),
            is_stale_sensor=is_stale,
            permit_type=permit_type,
        )

    return zone_states


# ---------------------------------------------------------------------------
# Standalone test — run this file directly to see the SVG output
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("VIGIL M6 - Plant Heatmap")
    print("Generating test SVG with simulated Vizag scenario state...")

    # Simulate the Vizag scenario at t=8min (C3 at CRITICAL)
    test_states = {
        "A1": ZoneState("A1", risk_score=12, severity="SAFE",  worker_count=2),
        "A2": ZoneState("A2", risk_score=0,  severity="SAFE",  worker_count=0),
        "B2": ZoneState("B2", risk_score=35, severity="LOW",   worker_count=3,
                        co_ppm=28.0),
        "C3": ZoneState("C3", risk_score=87, severity="CRITICAL", worker_count=6,
                        co_ppm=87.0, has_hot_work_permit=True, permit_type="hot_work"),
        "D4": ZoneState("D4", risk_score=0,  severity="SAFE",  worker_count=1),
        "E5": ZoneState("E5", risk_score=0,  severity="SAFE",  worker_count=2),
    }

    svg = generate_plant_svg(test_states)

    # Write to file for inspection
    out_path = "test_heatmap.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><body style='background:#000'>{svg}</body></html>")

    print(f"SVG written to {out_path} - open in browser to inspect")
    print(f"SVG length: {len(svg)} chars")
    print("Zone C3 should appear dark red/pulsing with HOT WORK PTW badge and 6 worker dots")