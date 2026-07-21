# -*- coding: utf-8 -*-
"""
VIGIL -- M7 Dashboard UI (v2)
==============================
Cleaner, tabbed layout. Less information on screen at once.
M9 CCTV Vision Module integrated.

LAYOUT
------
Header + critical banner (full width)
  Sidebar: controls + live zone tiles
  Main area -- 3 tabs:
    [Risk Assessment] : big score + explanation + rules + counterfactual
    [Permit Watch]    : active permits + live conflicts (M5)
    [CCTV Vision]     : zone camera feeds + PPE status (M9)
  Below tabs:
    [Past Incidents]  : RAG matches (M4)
    [Response]        : M8 emergency report (shown only when CRITICAL fires)
    [Log]             : notification log

THREADING MODEL (unchanged from v1)
-------------------------------------
VigilPipeline lives in st.session_state across Streamlit reruns.
M1-M9 all run in daemon threads. UI polls get_state() on each rerun.
"""

import base64
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Path setup -- make all modules importable
# ---------------------------------------------------------------------------
VIGIL_ROOT = Path(__file__).parent.parent
for _mod in [
    "m1_sensor_simulator", "m2_data_fusion", "m3_risk_engine",
    "m4_rag_incident_memory", "m5_permit_watch", "m6_plant_heatmap",
    "m8_response_orchestrator", "m9_cctv_vision",
]:
    sys.path.insert(0, str(VIGIL_ROOT / _mod))

from sensor_simulator import SensorSimulator
from data_fusion import DataFusionLayer
from risk_engine import CompoundRiskEngine, RiskEvent
from rag_incident_memory import RAGIncidentMemory
from permit_watch import PermitWatch, PermitConflict
from plant_heatmap import generate_plant_svg, build_zone_states_from_pipeline
from response_orchestrator import ResponseOrchestrator
from cctv_vision import CCTVVisionModule, CCTVReading

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VIGIL -- Industrial Safety Intelligence",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stAppDeployButton {display:none;}
    #MainMenu {visibility:hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    div[data-testid="stTabs"] button {font-size: 14px; font-weight: 500;}
    .vigil-score {font-size:72px; font-weight:800; line-height:1; text-align:center;}
    .vigil-badge {
        display:inline-block; padding:4px 16px; border-radius:12px;
        font-weight:700; letter-spacing:1px; font-size:13px;
        color:#fff; text-align:center;
    }
    .vigil-card {
        background:#161616; border-radius:8px; padding:12px 16px;
        margin-bottom:10px;
    }
    .vigil-bar {border-left:3px solid; padding:10px 14px;
        border-radius:0 6px 6px 0; background:#161616; margin-bottom:8px;}
    .vigil-rule-row {
        display:flex; align-items:center; gap:8px; padding:4px 0;
        border-bottom:1px solid #222;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
COLORS = {
    "SAFE":     "#2ECC71",
    "LOW":      "#F4D03F",
    "MEDIUM":   "#E67E22",
    "HIGH":     "#E74C3C",
    "CRITICAL": "#8B0000",
}


def score_color(score: int) -> str:
    if score >= 85: return COLORS["CRITICAL"]
    if score >= 70: return COLORS["HIGH"]
    if score >= 45: return COLORS["MEDIUM"]
    if score >= 20: return COLORS["LOW"]
    return COLORS["SAFE"]


def severity_color(sev: str) -> str:
    return COLORS.get(sev, "#555")


ZONES = ["A1", "B2", "C3", "D4"]
FUSION_INTERVAL = 8
TICK_INTERVAL = 3.0  # 3s per tick = gradual escalation over ~4 minutes
MAX_LOG = 100
MAX_FEED = 15


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class VigilPipeline:
    def __init__(self, scenario: str = "vizag"):
        self.scenario = scenario
        self.lock = threading.Lock()

        # State buckets
        self.zone_readings: dict = {}
        self.risk_events: dict = {}
        self.alert_feed: deque = deque(maxlen=MAX_FEED)
        self.log: deque = deque(maxlen=MAX_LOG)
        self.score_history: dict = {}
        self.rag_context: dict = {}
        self.permit_status: dict = {}
        self.permit_conflicts: dict = {}
        self.conflict_feed: deque = deque(maxlen=MAX_FEED)
        self.cctv_readings: dict = {}          # zone -> CCTVReading dict
        self.active_response: Optional[dict] = None

        self.ticks = 0
        self.snapshots = 0
        self.events = 0
        self.conflicts = 0

        # Build modules
        self.rag = RAGIncidentMemory(
            data_dir=VIGIL_ROOT / "m4_rag_incident_memory" / "data"
        )
        self.permit_watch = PermitWatch(
            data_dir=VIGIL_ROOT / "m5_permit_watch" / "data"
        )
        self.orchestrator = ResponseOrchestrator(
            reports_dir=VIGIL_ROOT / "m8_response_orchestrator" / "reports",
            claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
            on_response_ready=self._on_response,
            critical_threshold=75,
        )
        self.engine = CompoundRiskEngine(
            claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
            cooldown_sec=15,
            min_score_to_alert=15,
        )
        self.fusion = DataFusionLayer(
            fusion_interval=FUSION_INTERVAL,
            data_dir=VIGIL_ROOT / "m2_data_fusion" / "data",
        )
        self.sim = SensorSimulator(
            scenario=scenario,
            tick_interval=TICK_INTERVAL,
            loop=True,
            enable_mqtt=False,
            add_noise=True,
            data_dir=VIGIL_ROOT / "m1_sensor_simulator" / "data",
        )
        self.cctv = CCTVVisionModule(
            scenario=scenario,
            poll_interval=10,
            zones=ZONES,
            data_dir=VIGIL_ROOT / "m9_cctv_vision" / "data" / "mock_frames",
        )

        # Wire callbacks
        self.sim.register_callback(self._on_tick)
        self.fusion.register_callback(self._on_snapshot)
        self.engine.register_callback(self._on_risk_event)
        self.permit_watch.register_callback(self._on_conflict)
        self.cctv.register_callback(self._on_cctv)

        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._log("VIGIL started", "SYSTEM")
        self.fusion.start()
        # Pre-warm M5
        for z in ZONES:
            self.permit_watch.ingest_sensor_reading({
                "zone": z, "co_ppm": 10.0, "ch4_percent_lel": 1.0,
                "oxygen_percent": 20.9, "h2s_ppm": 0.5,
                "permit_active": False, "permit_type": "none",
                "shift_changeover_in_min": 60, "tick": 0,
            })
        self._refresh_permits()
        self.sim.start_async()
        self.cctv.start()         # M9 starts independently

    def stop(self):
        self._running = False
        self.sim.stop()
        self.fusion.stop()
        self.cctv.stop()
        self._log("VIGIL stopped", "SYSTEM")

    # ---- callbacks ----
    def _on_tick(self, reading):
        d = reading.to_dict() if hasattr(reading, "to_dict") else reading
        with self.lock:
            self.ticks += 1
            self.zone_readings[d["zone"]] = d
        self.fusion.ingest_sensor_reading(reading)
        try:
            self.permit_watch.ingest_sensor_reading(reading)
            self._refresh_permits()
        except Exception as e:
            self._log(f"M5 error: {e}", "M5-ERR")

    def _on_snapshot(self, snapshot):
        with self.lock:
            self.snapshots += 1
        self._log(f"Snapshot {snapshot.snapshot_id}", "M2")
        self.engine.evaluate_snapshot(snapshot)

    def _on_risk_event(self, event: RiskEvent):
        with self.lock:
            self.events += 1
            self.risk_events[event.zone] = event.to_dict()
            self.score_history.setdefault(event.zone, deque(maxlen=40)).append(
                (event.evaluated_at, event.risk_score)
            )
            self.alert_feed.appendleft(event.to_dict())
        self._log(
            f"Zone {event.zone} score={event.risk_score} {event.severity}", "M3"
        )
        try:
            result = self.rag.retrieve_from_risk_event(event)
            with self.lock:
                self.rag_context[event.zone] = result.to_dict()
        except Exception as e:
            self._log(f"RAG error: {e}", "M4-ERR")
        try:
            self.orchestrator.handle_risk_event(event)
        except Exception as e:
            self._log(f"M8 error: {e}", "M8-ERR")

    def _on_response(self, result):
        with self.lock:
            self.active_response = result.to_dict()
        self._log(f"Response {result.response_id} ready", "M8")

    def _on_conflict(self, conflict: PermitConflict):
        with self.lock:
            self.conflicts += 1
            self.conflict_feed.appendleft(conflict.to_dict())
        self._log(
            f"Permit conflict {conflict.permit_id} zone {conflict.zone}: "
            f"{conflict.conflict_type}", "M5"
        )

    def _on_cctv(self, reading: CCTVReading):
        with self.lock:
            self.cctv_readings[reading.zone] = reading.to_dict()
        if reading.violation_count > 0:
            self._log(
                f"CCTV Zone {reading.zone}: {reading.violation_count} PPE violation(s)", "M9"
            )

    def _refresh_permits(self):
        try:
            ps, pc = {}, {}
            for z in ZONES:
                pc[z] = [c.to_dict() for c in self.permit_watch.get_conflicts(z)]
                ps[z] = [
                    s.__dict__ if hasattr(s, "__dict__") else s
                    for s in self.permit_watch.get_permit_status(z)
                ]
            with self.lock:
                self.permit_conflicts = pc
                self.permit_status = ps
        except Exception:
            pass

    def _log(self, msg: str, src: str):
        with self.lock:
            self.log.appendleft({
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "source": src,
                "message": msg,
            })

    def get_state(self) -> dict:
        with self.lock:
            return {
                "zone_readings": dict(self.zone_readings),
                "risk_events": dict(self.risk_events),
                "alert_feed": list(self.alert_feed),
                "log": list(self.log),
                "score_history": {z: list(h) for z, h in self.score_history.items()},
                "rag_context": dict(self.rag_context),
                "permit_status": dict(self.permit_status),
                "permit_conflicts": dict(self.permit_conflicts),
                "conflict_feed": list(self.conflict_feed),
                "cctv_readings": dict(self.cctv_readings),
                "active_response": self.active_response,
                "ticks": self.ticks,
                "snapshots": self.snapshots,
                "events": self.events,
                "conflicts": self.conflicts,
            }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for _k, _v in [
    ("pipeline", None), ("running", False), ("scenario", "vizag")
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Helper: highest-risk zone
# ---------------------------------------------------------------------------
def top_zone(state: dict) -> tuple[str, str, int]:
    """Returns (zone_id, severity, score) of the worst active zone."""
    rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "SAFE": 0}
    best = ("--", "SAFE", 0)
    for z, ev in state["risk_events"].items():
        if rank.get(ev["severity"], 0) > rank.get(best[1], 0):
            best = (z, ev["severity"], ev["risk_score"])
    return best


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## VIGIL")
    st.caption("Industrial Safety Intelligence")
    st.divider()

    _scenario_opts = ["multizone", "vizag", "normal", "gas_leak", "confined_space"]
    scenario = st.selectbox(
        "Scenario",
        _scenario_opts,
        index=_scenario_opts.index(st.session_state.scenario),
        help="multizone = compound risk across several zones at once. "
             "vizag = full compound risk reconstruction. normal = safe baseline. "
             "gas_leak = single-sensor only (proves VIGIL doesn't over-alert). "
             "confined_space = oxygen depletion scenario.",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start", use_container_width=True, type="primary"):
            if (st.session_state.pipeline is None
                    or scenario != st.session_state.scenario):
                if st.session_state.pipeline:
                    st.session_state.pipeline.stop()
                st.session_state.pipeline = VigilPipeline(scenario=scenario)
                st.session_state.scenario = scenario
            st.session_state.pipeline.start()
            st.session_state.running = True
    with c2:
        if st.button("Stop", use_container_width=True):
            if st.session_state.pipeline:
                st.session_state.pipeline.stop()
            st.session_state.running = False

    st.divider()

    # Live zone tiles -- compact, sidebar width
    if st.session_state.pipeline:
        state = st.session_state.pipeline.get_state()
        for z in ZONES:
            ev = state["risk_events"].get(z)
            score = ev["risk_score"] if ev else 0
            sev = ev["severity"] if ev else "SAFE"
            col = score_color(score)
            reading = state["zone_readings"].get(z, {})
            co = reading.get("co_ppm")
            co_str = f"CO {co:.0f}ppm" if co is not None else "no sensor"
            cctv_r = state["cctv_readings"].get(z, {})
            ppe_flag = ""
            if cctv_r.get("violation_count", 0) > 0:
                ppe_flag = " | PPE!"

            st.markdown(
                f"""<div style="border-left:4px solid {col}; padding:6px 10px;
                margin-bottom:6px; background:#111; border-radius:0 6px 6px 0;">
                <div style="display:flex;justify-content:space-between;">
                  <span style="font-size:13px;font-weight:600;color:#eee;">Zone {z}</span>
                  <span style="font-size:12px;font-weight:700;color:{col};">{score}/100</span>
                </div>
                <div style="font-size:11px;color:#888;">{co_str}{ppe_flag}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # Alert feed -- chronological, cross-zone. Dropped in v2; restored here
    # in a compact sidebar form so the tabbed main area stays uncluttered.
    st.markdown("**Alert Feed**")
    if st.session_state.pipeline:
        _feed = state["alert_feed"]
        if not _feed:
            st.caption("No alerts yet.")
        else:
            for _a in _feed[:6]:
                _border = severity_color(_a["severity"])
                _ts = _a["evaluated_at"].split("T")[1][:8] if "T" in _a["evaluated_at"] else ""
                st.markdown(
                    f"""<div class="vigil-bar" style="border-color:{_border};
                    padding:6px 10px;margin-bottom:6px;">
                    <div style="font-size:11px;color:#888;">{_ts} &middot; Zone {_a['zone']}</div>
                    <div style="font-size:12px;color:#ddd;">
                    Score {_a['risk_score']} -- {_a['severity']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.divider()

    with st.expander("System Status"):
        if st.session_state.pipeline:
            s = st.session_state.pipeline.get_state()
            st.caption(f"Ticks: {s['ticks']}")
            st.caption(f"Snapshots: {s['snapshots']}")
            st.caption(f"Risk events: {s['events']}")
            st.caption(f"Permit conflicts: {s['conflicts']}")

    auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)
    st.caption("M1 Sensor > M2 Fusion > M3 Risk Engine > M4 RAG\nM1 > M5 Permit Watch | M3 > M8 Orchestrator\nM9 CCTV Vision (independent)")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
# Header
st.markdown(
    "<h2 style='margin:0 0 4px;'>VIGIL -- Industrial Safety Intelligence</h2>"
    "<p style='color:#888;margin:0 0 12px;'>Real-time compound risk detection "
    "for zero-harm plant operations</p>",
    unsafe_allow_html=True,
)

if st.session_state.pipeline is None:
    st.info(
        "Select a scenario and click **Start** in the sidebar.  \n"
        "**vizag** reconstructs the compound risk pattern that preceded the "
        "Visakhapatnam Steel Plant explosion (January 2025, 8 fatalities)."
    )
    st.stop()

state = st.session_state.pipeline.get_state()

# ---------------------------------------------------------------------------
# CRITICAL banner -- only when M8 fires
# ---------------------------------------------------------------------------
resp = state.get("active_response")
if resp and resp.get("success"):
    tz = resp.get("zone", "?")
    sc = resp.get("risk_score", 0)
    st.markdown(
        f"""<div style="background:#3a0000;border:2px solid #8B0000;
        border-radius:8px;padding:14px 20px;margin-bottom:12px;">
        <span style="font-size:18px;font-weight:800;color:#ff4444;">
        CRITICAL ALERT -- ZONE {tz} -- {sc}/100</span>
        <span style="color:#ffaaaa;font-size:13px;margin-left:16px;">
        Emergency sequence initiated. Report generated.</span>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_risk, tab_permit, tab_cctv, tab_log = st.tabs([
    "Risk Assessment", "Permit Watch (M5)", "CCTV Vision (M9)", "Log"
])


# ===========================================================================
# TAB 1: RISK ASSESSMENT
# ===========================================================================
with tab_risk:
    zone_id, zone_sev, zone_score = top_zone(state)
    ev = state["risk_events"].get(zone_id)

    col_score, col_detail = st.columns([1, 2], gap="large")

    with col_score:
        st.markdown("**Plant Zone Map**")
        try:
            zone_states_m6 = build_zone_states_from_pipeline(state)
            plant_html = generate_plant_svg(zone_states_m6, pulse_critical=True)
            components.html(plant_html, height=280, scrolling=False)
            st.caption(
                "Zones colored by compound risk score (0-100). "
                "Orange dashed border = active PTW permit. "
                "Worker dots pulse red when zone is CRITICAL."
            )
        except Exception:
            st.caption("Plant heatmap unavailable")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Risk score
        col = score_color(zone_score)
        st.markdown(
            f"<div class='vigil-score' style='color:{col};'>{zone_score}"
            f"<span style='font-size:24px;color:#555;'>/100</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:center;margin:6px 0 12px;'>"
            f"<span class='vigil-badge' style='background:{col};'>"
            f"{zone_sev} -- ZONE {zone_id}</span></div>",
            unsafe_allow_html=True,
        )

        # Score trend chart -- this is the moment judges watch the number
        # climb live, so it gets real vertical space and labeled axes
        # rather than being squeezed into a tiny inset.
        if any(state["score_history"].values()):
            st.markdown("**Risk score trend (live)**")
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                for z, hist in state["score_history"].items():
                    ys = [h[1] for h in hist]
                    fig.add_trace(go.Scatter(
                        y=ys, mode="lines+markers", name=f"Zone {z}",
                        line=dict(width=3, color=score_color(ys[-1])),
                        marker=dict(size=7),
                    ))
                fig.update_layout(
                    height=340, margin=dict(l=10, r=10, t=10, b=30),
                    template="plotly_dark", showlegend=True,
                    yaxis=dict(
                        range=[0, 100], title="Risk score",
                        gridcolor="rgba(255,255,255,0.12)", showgrid=True,
                        dtick=20,
                    ),
                    xaxis=dict(
                        title="Evaluation #",
                        gridcolor="rgba(255,255,255,0.06)", showgrid=True,
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.caption("Install plotly for the trend chart: pip install plotly")

    with col_detail:
        if not ev:
            st.success("No compound risk conditions detected. All zones nominal.")
        else:
            col = score_color(zone_score)

            # Explanation
            explanation = str(ev.get("llm_explanation", "")).replace(
                " in None minutes", ""
            ).replace("None minutes", "recently").replace("in None", "recently")
            st.markdown(
                f"""<div class="vigil-bar" style="border-color:{col};font-size:14px;
                line-height:1.7;">{explanation}</div>""",
                unsafe_allow_html=True,
            )

            # Predicted time
            ptc = ev.get("predicted_minutes_to_critical")
            if ptc:
                st.warning(f"Predicted time to critical: **{ptc} minutes**")

            # Rules fired -- with running total
            st.markdown("**Rules fired**")
            total = 0
            parts = []
            for r in ev["rules_fired"]:
                c = r["score_contribution"]
                total += c
                parts.append(str(c))
                rule_col = score_color(min(c * 3, 100))
                st.markdown(
                    f"""<div class="vigil-rule-row">
                    <span style="font-size:11px;color:#888;width:56px;flex-shrink:0;">
                    {r['rule_id']}</span>
                    <span style="font-size:13px;color:#ddd;flex:1;">{r['name']}</span>
                    <span style="font-size:12px;font-weight:700;color:{rule_col};">
                    +{c}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            capped = " (capped)" if total > 100 else ""
            st.caption(
                f"{' + '.join(parts)} = **{min(total,100)}/100**{capped} "
                f"-- transparent, auditable arithmetic"
            )

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # Counterfactual
            cf = str(ev.get("counterfactual") or "").replace("None", "--")
            if cf and cf != "--":
                st.markdown(
                    f"""<div style="background:#0d1820;border:1px solid #1e3a50;
                    border-radius:6px;padding:10px 14px;font-size:13px;
                    color:#a8d4f0;line-height:1.6;margin-bottom:10px;">
                    Counterfactual: {cf}</div>""",
                    unsafe_allow_html=True,
                )

            # Actions
            actions = ev.get("recommended_actions", [])
            if actions:
                st.markdown("**Recommended actions**")
                for i, a in enumerate(actions, 1):
                    st.markdown(f"{i}. {a}")

            # OISD clauses
            oisd = ev.get("oisd_clauses", [])
            if oisd:
                with st.expander("OISD / regulatory clauses cited", expanded=False):
                    for clause in oisd:
                        st.markdown(f"- {clause}")

    # Past incidents (M4 RAG) -- below the two columns
    rag = state["rag_context"].get(zone_id)
    if rag and rag.get("top_incidents"):
        st.divider()
        st.markdown("**Similar past incidents (M4 RAG)**")
        st.caption(rag.get("headline_match", ""))
        inc_cols = st.columns(min(len(rag["top_incidents"]), 3))
        for col_i, inc in enumerate(rag["top_incidents"][:3]):
            with inc_cols[col_i]:
                pct = inc.get("match_percent", 0)
                lbl = inc.get("match_label", "")
                fat = inc.get("fatalities", 0)
                fat_str = f"{fat} fatalities" if fat else "no fatalities"
                border_c = "#8B0000" if fat > 0 else "#444"
                st.markdown(
                    f"""<div style="border:1px solid {border_c};border-radius:8px;
                    padding:12px;background:#111;height:100%;">
                    <div style="font-size:12px;font-weight:700;color:{border_c};">
                    {pct}% match -- {lbl}</div>
                    <div style="font-size:13px;font-weight:600;color:#ddd;
                    margin:4px 0;">{inc['title']}</div>
                    <div style="font-size:11px;color:#888;">{inc['facility']}</div>
                    <div style="font-size:11px;color:#aaa;margin-top:4px;">
                    {fat_str}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                with st.expander("Details", expanded=False):
                    st.markdown(f"**Summary:** {inc.get('summary','')}")
                    root_causes = inc.get("root_causes", [])
                    if root_causes:
                        st.markdown("**Root causes:**")
                        for cause in root_causes:
                            st.markdown(f"- {cause}")

    # M8 response report -- only when fired
    if resp and resp.get("success"):
        st.divider()
        with st.expander("Emergency Response Report (M8)", expanded=True):
            alerts = resp.get("alert_messages", [])
            if alerts:
                st.markdown(f"**{len(alerts)} notifications dispatched:**")
                for a in alerts:
                    icon = {"sms": "\U0001F4F1", "whatsapp": "\U0001F4AC",
                            "scada_log": "\U0001F5A5\uFE0F", "dashboard": "\U0001F4FA"}.get(
                        a.get("channel", ""), "\U0001F4E3"
                    )
                    st.markdown(
                        f"{icon} **{a.get('recipient','')}**  \n"
                        f"*{a.get('message_en','')}*"
                    )
                    if a.get("message_hi"):
                        st.caption(f"Hindi: {a.get('message_hi','')}")
                    st.divider()

            report = resp.get("report")
            if report:
                st.markdown("**DGFASLI Preliminary Incident Report (AI-generated)**")
                st.markdown(
                    f"""
                    <div style="
                        background:#0d1a0d; border:1px solid #2a5a2a;
                        border-radius:8px; padding:16px; font-family:monospace;
                        font-size:12px; color:#90ee90; white-space:pre-wrap;
                        max-height:500px; overflow-y:auto;
                    ">
                    {report.get('incident_summary','No summary generated')}

IMMEDIATE ACTIONS:
{chr(10).join(f"  {i+1}. {a}" for i, a in enumerate(report.get('immediate_actions', [])))}

REGULATORY CLAUSES:
{chr(10).join(f"  - {c}" for c in report.get('regulatory_violations', []))}

PRELIMINARY ROOT CAUSE:
{report.get('preliminary_root_cause', '')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Report ID: {report.get('report_id','')} | "
                    f"Generated: {report.get('generated_at','')} | "
                    f"Type: {report.get('report_type','')}"
                )


# ===========================================================================
# TAB 2: PERMIT WATCH (M5)
# ===========================================================================
with tab_permit:
    st.caption(
        "Every PTW is checked against the gas/O2 thresholds recorded at issuance -- "
        "not generic limits. CO was 12 ppm when this permit was signed. "
        "It conflicts the moment CO passes that permit's own threshold."
    )

    all_conflicts = [
        c for cl in state["permit_conflicts"].values() for c in cl
    ]
    if all_conflicts:
        st.error(
            f"{len(all_conflicts)} permit(s) currently in conflict with live sensor conditions"
        )

    col_p, col_c = st.columns([3, 2], gap="medium")

    with col_p:
        st.markdown("**Active permits**")
        any_p = False
        for z in ZONES:
            for p in state["permit_status"].get(z, []):
                any_p = True
                sev = p.get("highest_conflict_severity", "NONE")
                col = COLORS.get(sev if sev != "NONE" else "SAFE", "#2ECC71")
                expiry = " -- expiring soon" if p.get("expiry_warning") else ""
                st.markdown(
                    f"""<div class="vigil-bar" style="border-color:{col};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                      <span style="font-size:13px;font-weight:600;color:#eee;">
                      {p.get('permit_id','')} &nbsp;
                      <span style="font-weight:400;color:#999;">
                      {p.get('permit_type','').replace('_',' ').title()}</span>
                      </span>
                      <span style="font-size:11px;color:{col};font-weight:700;">
                      Zone {p.get('zone','')} &nbsp;|&nbsp; {sev}</span>
                    </div>
                    <div style="font-size:12px;color:#999;margin-top:3px;">
                    {p.get('issued_to','')} -- {p.get('work_description','')}</div>
                    <div style="font-size:11px;color:#666;">
                    {p.get('valid_from','')}--{p.get('valid_until','')}
                    &nbsp;|&nbsp; {p.get('conflict_count',0)} conflict(s){expiry}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        if not any_p:
            st.caption("No active permits for tracked zones yet.")

    with col_c:
        st.markdown("**Conflict feed**")
        if not state["conflict_feed"]:
            st.caption("No conflicts detected yet.")
        else:
            for c in state["conflict_feed"][:6]:
                sev_c = (
                    "CRITICAL" if c["conflict_type"] in
                    ("OXYGEN_DEPLETION", "HOT_WORK_NO_FIRE_WATCH")
                    else "HIGH"
                )
                border = COLORS.get(sev_c, "#444")
                ts = c["detected_at"].split("T")[1][:8] if "T" in c["detected_at"] else ""
                with st.expander(
                    f"{ts}  {c['permit_id']}  "
                    f"{c['conflict_type'].replace('_',' ').title()}"
                ):
                    st.markdown(
                        f"""<div style="border-left:3px solid {border};
                        padding:8px 12px;background:#111;border-radius:0 6px 6px 0;
                        font-size:13px;color:#ccc;">{c['description']}</div>""",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Threshold `{c['permit_threshold_breached']}` = "
                        f"{c['threshold_value']} | actual = {c['actual_value']} | "
                        f"action: **{c['action_required']}** | {c['oisd_clause']}"
                    )


# ===========================================================================
# TAB 3: CCTV VISION (M9)
# ===========================================================================
with tab_cctv:
    cctv_data = state["cctv_readings"]
    backend_used = next(
        (r.get("backend", "--") for r in cctv_data.values()), "--"
    )

    st.caption(
        f"PPE detection backend: **{backend_used}**  |  "
        "Workers without helmets or safety vests flagged as violations.  |  "
        "In production: plug live RTSP feeds here."
    )

    total_workers = sum(r.get("workers_detected", 0) for r in cctv_data.values())
    total_viol = sum(r.get("violation_count", 0) for r in cctv_data.values())
    compliant = total_workers - total_viol
    rate = (compliant / total_workers * 100) if total_workers > 0 else 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Workers detected", total_workers)
    m2.metric("PPE compliant", compliant)
    m3.metric("Compliance rate", f"{rate:.0f}%",
              delta=f"-{total_viol} violation(s)" if total_viol else "All clear")

    st.divider()

    if not cctv_data:
        st.info("CCTV module starting -- first scan in ~10 seconds after Start.")
    else:
        # 2x2 grid of zone camera views
        row1 = st.columns(2, gap="medium")
        row2 = st.columns(2, gap="medium")
        grid = [row1[0], row1[1], row2[0], row2[1]]

        for col_i, z in enumerate(ZONES):
            r = cctv_data.get(z, {})
            with grid[col_i]:
                workers = r.get("workers_detected", 0)
                viols = r.get("violation_count", 0)
                comp_rate = r.get("compliance_rate", 1.0)
                status = r.get("backend", "--")
                sev_label = r.get("severity") if hasattr(r, "get") else (
                    "COMPLIANT" if viols == 0 else
                    "WARNING" if viols == 1 else "VIOLATION"
                )
                if viols == 0:
                    sev_label = "COMPLIANT"
                elif viols == 1:
                    sev_label = "WARNING"
                else:
                    sev_label = "VIOLATION"

                border_col = (
                    COLORS["SAFE"] if viols == 0
                    else COLORS["MEDIUM"] if viols == 1
                    else COLORS["HIGH"]
                )

                # Camera frame image
                b64 = r.get("annotated_frame_b64")
                if b64:
                    st.markdown(
                        f"<img src='data:image/jpeg;base64,{b64}' "
                        f"style='width:100%;border-radius:6px;"
                        f"border:2px solid {border_col};margin-bottom:6px;'>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Placeholder when no frame yet
                    st.markdown(
                        f"""<div style="background:#111;border:2px solid {border_col};
                        border-radius:6px;height:140px;display:flex;align-items:center;
                        justify-content:center;color:#444;font-size:13px;
                        margin-bottom:6px;">Zone {z} -- no frame yet</div>""",
                        unsafe_allow_html=True,
                    )

                # Zone summary
                st.markdown(
                    f"""<div style="border-left:3px solid {border_col};
                    padding:6px 10px;background:#111;border-radius:0 6px 6px 0;">
                    <div style="display:flex;justify-content:space-between;">
                      <span style="font-size:13px;font-weight:600;color:#eee;">
                      Zone {z}</span>
                      <span style="font-size:12px;font-weight:700;color:{border_col};">
                      {sev_label}</span>
                    </div>
                    <div style="font-size:11px;color:#888;">
                    {workers} worker(s) detected &nbsp;|&nbsp;
                    {viols} violation(s) &nbsp;|&nbsp;
                    {comp_rate:.0%} compliant</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Violations detail
                violations = r.get("violations", [])
                for v in violations:
                    wid = v.get("worker_id", "?")
                    missing = v.get("missing_ppe", [])
                    conf = v.get("confidence", 0)
                    st.markdown(
                        f"""<div style="background:#2a0000;border-radius:4px;
                        padding:4px 8px;margin-top:4px;font-size:11px;color:#ffaaaa;">
                        Worker {wid}: missing {', '.join(missing)}
                        (conf {conf:.0%})</div>""",
                        unsafe_allow_html=True,
                    )


# ===========================================================================
# TAB 4: LOG
# ===========================================================================
with tab_log:
    log = state["log"]
    if log:
        import pandas as pd
        df = pd.DataFrame(log)
        st.dataframe(
            df,
            use_container_width=True,
            height=500,
            hide_index=True,
            column_config={
                "time": st.column_config.TextColumn("Time", width="small"),
                "source": st.column_config.TextColumn("Module", width="small"),
                "message": st.column_config.TextColumn("Event"),
            },
        )
    else:
        st.caption("No events logged yet.")


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if auto_refresh and st.session_state.running:
    time.sleep(2)
    st.rerun()
