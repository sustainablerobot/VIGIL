# -*- coding: utf-8 -*-
"""
VIGIL — M7 Dashboard UI
=========================
The mission control screen. This is what the safety officer stares at.

Layout:
  LEFT PANEL   : Plant map / zone grid with live risk-colored tiles
  CENTER       : Current risk score (big number) + plain-language explanation
  RIGHT PANEL  : Alert history feed + similar past incidents (from M4 RAG)
  BOTTOM       : Notification log (every event, scrollable, timestamped)

Everything updates live as M1 (sensor simulator) plays out a scenario,
flows through M2 (data fusion), gets scored by M3 (risk engine), and
M4 (RAG) attaches historical context.

HOW THIS MODULE WORKS
-----------------------
1. On "Start Simulation", spins up M1 SensorSimulator in a background thread
2. M1 feeds M2 DataFusionLayer via registered callback
3. M2 fires a fusion cycle every N seconds, feeding M3 CompoundRiskEngine
4. M3 evaluates compound rules, fires RiskEvents above threshold
5. Every RiskEvent triggers an M4 RAG lookup for similar historical incidents
6. All of this lands in Streamlit's session_state, which the UI polls and renders
7. st.autorefresh-style polling redraws the screen every 2 seconds without
   the user needing to manually refresh

THREADING MODEL
----------------
Streamlit reruns the whole script top-to-bottom on every interaction/refresh.
Background simulation threads (M1/M2 timers) must NOT be recreated on every
rerun, or you'd spawn duplicate simulators. This is solved by:
  - Storing the running pipeline object in st.session_state (persists across reruns)
  - A guard flag (pipeline_started) so Start is only wired once per session
  - All cross-thread data (events, snapshots) appended to thread-safe lists
    that the main Streamlit thread reads from on each rerun

WHY STREAMLIT FOR THIS DEMO
-----------------------------
Streamlit lets one person build a real-time multi-panel ops dashboard in a
single Python file with no separate frontend build step — critical for a
solo 2-4 week build. In production this would likely become a proper
React/Next.js app fed by the same M1-M4 backend over WebSocket, but for the
hackathon demo, Streamlit gets you 90% of the visual impact for 10% of the time.

TECHNOLOGIES
------------
- streamlit: the UI framework itself
- plotly: zone risk gauge + historical score timeline chart
- threading: background pipeline execution without blocking the UI thread
- pandas: notification log table formatting
"""

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
# Make M1-M8 importable regardless of where streamlit run is invoked from
# ---------------------------------------------------------------------------
VIGIL_ROOT = Path(__file__).parent.parent
for module_dir in [
    "m1_sensor_simulator", "m2_data_fusion", "m3_risk_engine",
    "m4_rag_incident_memory", "m5_permit_watch", "m6_plant_heatmap",
    "m8_response_orchestrator",
]:
    sys.path.insert(0, str(VIGIL_ROOT / module_dir))

from sensor_simulator import SensorSimulator
from data_fusion import DataFusionLayer
from risk_engine import CompoundRiskEngine, RiskEvent
from rag_incident_memory import RAGIncidentMemory
from permit_watch import PermitWatch, PermitConflict
from plant_heatmap import generate_plant_svg, build_zone_states_from_pipeline
from response_orchestrator import ResponseOrchestrator

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VIGIL — Industrial Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit auto-injects a "Deploy" button into the top-right toolbar on every
# app. It does nothing meaningful in this local-run context and is an inert
# button a judge could click during Q&A — hide it explicitly.
st.markdown(
    """
    <style>
        .stAppDeployButton {display: none;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 4-tier gradient (score-based, not just severity label) so a 35 visually
# reads as calmer than a 75 even though both are "not safe." Severity label
# still drives the badge text; this drives the badge/border color, derived
# directly from score so the gradient is continuous and judges see escalation.
SEVERITY_COLORS = {
    "SAFE":     "#2ECC71",   # green   (0-20)
    "LOW":      "#F4D03F",   # yellow  (20-45)
    "MEDIUM":   "#E67E22",   # orange  (45-70)
    "HIGH":     "#E74C3C",   # red     (70-85)
    "CRITICAL": "#8B0000",   # dark red (85-100)
}
SEVERITY_BG = {
    "SAFE":     "#1a3a2a",
    "LOW":      "#3a3520",
    "MEDIUM":   "#3a2a15",
    "HIGH":     "#3a1818",
    "CRITICAL": "#4a0a0a",
}
ZONE_IDS = ["A1", "B2", "C3", "D4"]
ZONE_LAYOUT = ZONE_IDS  # alias

def score_to_tier_color(score: int) -> str:
    """
    4-tier color purely from numeric score, independent of M3's severity label.
    Use this for any UI element where visual escalation matters more than
    matching M3's exact band boundaries (which are tuned for alerting logic,
    not necessarily for visual contrast).
    """
    if score >= 70:
        return SEVERITY_COLORS["CRITICAL"] if score >= 85 else SEVERITY_COLORS["HIGH"]
    if score >= 45:
        return SEVERITY_COLORS["MEDIUM"]
    if score >= 20:
        return SEVERITY_COLORS["LOW"]
    return SEVERITY_COLORS["SAFE"]
ZONE_LAYOUT = ["A1", "B2", "C3", "D4"]  # plant zone grid positions
MAX_LOG_ENTRIES = 100
MAX_ALERT_FEED = 15
FUSION_INTERVAL = 8     # seconds — fast enough for live demo, not 30s
TICK_INTERVAL = 1.0     # seconds per sensor reading


# ---------------------------------------------------------------------------
# Pipeline orchestration — wires M1 -> M2 -> M3 -> M4 together
# ---------------------------------------------------------------------------
class VigilPipeline:
    """
    Owns the M1-M4 instances and wires their callbacks together.
    Stored once in st.session_state so it survives Streamlit reruns.
    """

    def __init__(self, scenario: str = "vizag"):
        self.scenario = scenario
        self.lock = threading.Lock()

        # Shared state the UI reads from (thread-safe via lock)
        self.latest_zone_states: dict = {}      # zone -> latest sensor dict
        self.latest_risk_events: dict = {}       # zone -> latest RiskEvent dict
        self.alert_feed: deque = deque(maxlen=MAX_ALERT_FEED)
        self.notification_log: deque = deque(maxlen=MAX_LOG_ENTRIES)
        self.score_history: dict = {}            # zone -> list of (timestamp, score)
        self.rag_context: dict = {}               # zone -> latest RetrievalResult dict

        # M5 permit watch state — dashboard polls this
        self.permit_status: dict = {}            # zone -> list of PermitStatus dicts
        self.permit_conflicts: dict = {}         # zone -> list of active PermitConflict dicts
        self.conflict_feed: deque = deque(maxlen=MAX_ALERT_FEED)
        self.conflict_count = 0

        self.sim_start_time = None
        self.tick_count = 0
        self.snapshot_count = 0
        self.event_count = 0

        # Shared M8 response state — dashboard polls this
        self.active_critical_response: Optional[dict] = None

        # Build the pipeline
        self.rag = RAGIncidentMemory(data_dir=VIGIL_ROOT / "m4_rag_incident_memory" / "data")
        self.permit_watch = PermitWatch(data_dir=VIGIL_ROOT / "m5_permit_watch" / "data")  # absolute path
        self.orchestrator = ResponseOrchestrator(
            reports_dir=VIGIL_ROOT / "m8_response_orchestrator" / "reports",
            claude_api_key=os.getenv("ANTHROPIC_API_KEY"),  # set in .env or system env
            on_response_ready=self._on_response_ready,
            critical_threshold=75,
        )
        self.engine = CompoundRiskEngine(
            claude_api_key=os.getenv("ANTHROPIC_API_KEY"),  # set in .env or system env
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

        # Wire callbacks: M1 -> M2 -> M3 -> (M4 + UI state)
        #                 M1 -> M5 (independent permit-conflict watch)
        self.sim.register_callback(self._on_sensor_reading)
        self.fusion.register_callback(self._on_snapshot)
        self.engine.register_callback(self._on_risk_event)
        self.permit_watch.register_callback(self._on_permit_conflict)

        self._running = False
        self._sim_thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self.sim_start_time = datetime.now(timezone.utc)
        self._log("VIGIL pipeline started", "SYSTEM")
        self.fusion.start()
        # Pre-warm M5 with safe baseline readings for all zones that have permits
        # Without this, permit status shows empty until sensor data arrives for that zone
        for zone_id in ZONE_IDS:
            baseline = {
                "zone": zone_id, "co_ppm": 10.0, "ch4_percent_lel": 1.0,
                "oxygen_percent": 20.9, "h2s_ppm": 0.5,
                "permit_active": True, "permit_type": "none",
                "shift_changeover_in_min": 60, "tick": 0,
            }
            try:
                self.permit_watch.ingest_sensor_reading(baseline)
            except Exception:
                pass
        self._update_permit_state()
        self._sim_thread = self.sim.start_async()

    def _update_permit_state(self):
        """Refresh permit status snapshot — called after each sensor reading."""
        try:
            by_zone: dict = {}
            statuses: dict = {}
            for zone_id in ZONE_IDS:
                conflicts = self.permit_watch.get_conflicts(zone_id)
                by_zone[zone_id] = [c.to_dict() for c in conflicts]
                statuses[zone_id] = [
                    s.__dict__ if hasattr(s, "__dict__") else s
                    for s in self.permit_watch.get_permit_status(zone_id)
                ]
            with self.lock:
                self.permit_conflicts = by_zone
                self.permit_status = statuses
        except Exception as e:
            self._log(f"Permit state refresh error: {e}", "M5-ERR")

    def stop(self):
        self._running = False
        self.sim.stop()
        self.fusion.stop()
        self._log("VIGIL pipeline stopped", "SYSTEM")

    def switch_scenario(self, new_scenario: str):
        """Stop current pipeline and rebuild with a new scenario."""
        self.stop()
        time.sleep(0.3)
        self.__init__(scenario=new_scenario)
        self.start()

    # ------------------------------------------------------------------
    # Callbacks — these run on M1's background thread
    # ------------------------------------------------------------------
    def _on_sensor_reading(self, reading):
        """M1 fires this every tick (1/sec). Feed to M2 + M5 + update zone state."""
        d = reading.to_dict() if hasattr(reading, "to_dict") else reading
        with self.lock:
            self.tick_count += 1
            self.latest_zone_states[d["zone"]] = d
        self.fusion.ingest_sensor_reading(reading)

        # M5: permit-vs-live-conditions check, independent of M3
        try:
            self.permit_watch.ingest_sensor_reading(reading)
            self._update_permit_state()
        except Exception as e:
            self._log(f"M5 permit watch error: {e}", "M5-ERROR")

    def _on_snapshot(self, snapshot):
        """M2 fires this every fusion_interval seconds. Feed to M3."""
        with self.lock:
            self.snapshot_count += 1
        self._log(f"Snapshot {snapshot.snapshot_id} fused", "M2")
        self.engine.evaluate_snapshot(snapshot)

    def _on_risk_event(self, event: RiskEvent):
        """M3 fires this when a zone's score crosses the alert threshold."""
        with self.lock:
            self.event_count += 1
            self.latest_risk_events[event.zone] = event.to_dict()

            hist = self.score_history.setdefault(event.zone, deque(maxlen=40))
            hist.append((event.evaluated_at, event.risk_score))

            self.alert_feed.appendleft(event.to_dict())

        self._log(
            f"Zone {event.zone}: score={event.risk_score} severity={event.severity} "
            f"rules={len(event.rules_fired)}",
            "M3",
        )

        # M4: fetch similar historical incidents for this event
        try:
            result = self.rag.retrieve_from_risk_event(event)
            with self.lock:
                self.rag_context[event.zone] = result.to_dict()
            self._log(
                f"RAG match for {event.zone}: {result.top_incidents[0]['title'] if result.top_incidents else 'none'}",
                "M4",
            )
        except Exception as e:
            self._log(f"RAG lookup failed: {e}", "M4-ERROR")

        # M8: trigger emergency response if CRITICAL
        try:
            self.orchestrator.handle_risk_event(event)
        except Exception as e:
            self._log(f"M8 orchestrator error: {e}", "M8-ERROR")

    def _on_response_ready(self, response_result):
        """M8 fires this when incident report is ready."""
        with self.lock:
            self.active_critical_response = response_result.to_dict()
        self._log(
            f"CRITICAL RESPONSE: {response_result.response_id} | "
            f"report={'yes' if response_result.report else 'no'} | "
            f"duration={response_result.duration_sec}s",
            "M8",
        )

    def _on_permit_conflict(self, conflict: PermitConflict):
        """
        M5 fires this the moment a live sensor reading breaches the
        threshold a permit was actually issued against (e.g. CO was 12ppm
        at issuance, is now 66ppm). This is the inter-agent signal M3's
        CR-001 pattern is built to detect independently — M5 gives the
        safety officer the direct, permit-level explanation.
        """
        with self.lock:
            self.conflict_count += 1
            self.conflict_feed.appendleft(conflict.to_dict())
        self._log(
            f"PERMIT CONFLICT {conflict.permit_id} (zone {conflict.zone}): "
            f"{conflict.conflict_type} — {conflict.action_required}",
            "M5",
        )

    def _log(self, message: str, source: str):
        with self.lock:
            self.notification_log.appendleft({
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "source": source,
                "message": message,
            })

    # ------------------------------------------------------------------
    # Thread-safe snapshot getters for the UI thread
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        with self.lock:
            return {
                "zone_states": dict(self.latest_zone_states),
                "risk_events": dict(self.latest_risk_events),
                "alert_feed": list(self.alert_feed),
                "notification_log": list(self.notification_log),
                "score_history": {z: list(h) for z, h in self.score_history.items()},
                "rag_context": dict(self.rag_context),
                "permit_status": dict(self.permit_status),
                "permit_conflicts": dict(self.permit_conflicts),
                "conflict_feed": list(self.conflict_feed),
                "tick_count": self.tick_count,
                "snapshot_count": self.snapshot_count,
                "event_count": self.event_count,
                "conflict_count": self.conflict_count,
                "active_critical_response": self.active_critical_response,
            }


# ---------------------------------------------------------------------------
# Session state initialisation — pipeline must survive Streamlit reruns
# ---------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "vizag"


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ VIGIL Control")
    st.caption("Industrial Safety Intelligence Platform")
    st.divider()

    scenario = st.selectbox(
        "Scenario",
        options=["multizone", "vizag", "normal", "gas_leak", "confined_space"],
        index=["multizone", "vizag", "normal", "gas_leak", "confined_space"].index(
            st.session_state.current_scenario
        ),
        help="vizag = full compound risk reconstruction. normal = safe baseline. "
             "gas_leak = single-sensor only (proves VIGIL doesn't over-alert). "
             "confined_space = oxygen depletion scenario.",
    )

    col_start, col_stop = st.columns(2)
    with col_start:
        start_clicked = st.button("▶ Start", use_container_width=True, type="primary")
    with col_stop:
        stop_clicked = st.button("■ Stop", use_container_width=True)

    if start_clicked:
        if (st.session_state.pipeline is None or
                scenario != st.session_state.current_scenario):
            if st.session_state.pipeline is not None:
                st.session_state.pipeline.stop()
            st.session_state.pipeline = VigilPipeline(scenario=scenario)
            st.session_state.current_scenario = scenario
        st.session_state.pipeline.start()
        st.session_state.pipeline_running = True

    if stop_clicked and st.session_state.pipeline is not None:
        st.session_state.pipeline.stop()
        st.session_state.pipeline_running = False

    st.divider()

    with st.expander("⚙️ System Status", expanded=False):
        if st.session_state.pipeline is not None:
            state = st.session_state.pipeline.get_state()
            st.caption(f"Sensor ticks: {state['tick_count']}")
            st.caption(f"Fusion snapshots: {state['snapshot_count']}")
            st.caption(f"Risk events fired: {state['event_count']}")
            st.caption(f"Permit conflicts: {state.get('conflict_count', 0)}")

    st.divider()
    st.caption(
        "Pipeline: M1 Sensor → M2 Fusion → M3 Risk Engine → M4 RAG Memory\n\n"
        "M1 → M5 Permit Watch (independent) · M3 → M8 Response Orchestrator · "
        "M2/M3 → M6 Plant Heatmap"
    )
    auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:4px;">
        <h1 style="margin:0;">🛡️ VIGIL</h1>
        <span style="color:#888; font-size:16px;">Compound Risk Intelligence for Zero-Harm Operations</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.pipeline is None:
    st.info(
        "👈 Select a scenario and click **Start** in the sidebar to begin live monitoring. "
        "The 'vizag' scenario reconstructs the compound risk pattern that preceded the "
        "Visakhapatnam Steel Plant explosion (January 2025)."
    )
    st.stop()

state = st.session_state.pipeline.get_state()

# ---------------------------------------------------------------------------
# M8 CRITICAL ALERT BANNER — shown when orchestrator fires
# ---------------------------------------------------------------------------
critical_response = state.get("active_critical_response")
if critical_response and critical_response.get("success"):
    report = critical_response.get("report")
    zone_cr = critical_response.get("zone", "?")
    score_cr = critical_response.get("risk_score", 0)
    alerts_cr = critical_response.get("alert_messages", [])

    st.markdown(
        f"""
        <div style="
            background:#4a0a0a; border:2px solid #8B0000;
            border-radius:10px; padding:16px 20px; margin-bottom:16px;
        ">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                <span style="font-size:28px;">🚨</span>
                <span style="font-size:20px; font-weight:800; color:#ff4444; letter-spacing:1px;">
                    CRITICAL ALERT — ZONE {zone_cr} — SCORE {score_cr}/100
                </span>
            </div>
            <div style="color:#ffaaaa; font-size:14px; margin-bottom:8px;">
                Emergency response sequence initiated.
                {len(alerts_cr)} notifications dispatched.
                DGFASLI preliminary incident report generated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show alert messages dispatched
    with st.expander("📡 Alert notifications dispatched", expanded=True):
        for alert in alerts_cr:
            channel_icon = {"sms": "📱", "whatsapp": "💬", "scada_log": "🖥️", "dashboard": "📺"}.get(
                alert.get("channel", ""), "📣"
            )
            st.markdown(
                f"{channel_icon} **{alert.get('recipient','')}**  \n"
                f"*{alert.get('message_en','')}*"
            )
            if alert.get("message_hi"):
                st.caption(f"Hindi: {alert.get('message_hi','')}")
            st.divider()

    # Show the full generated incident report
    if report:
        with st.expander("📄 DGFASLI Preliminary Incident Report (AI-generated)", expanded=True):
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

    st.divider()


# ---------------------------------------------------------------------------
# Helper: determine each zone's current severity for the heatmap
# ---------------------------------------------------------------------------
def zone_severity(zone_id: str) -> tuple[str, int]:
    event = state["risk_events"].get(zone_id)
    if event:
        return event["severity"], event["risk_score"]
    return "SAFE", 0


# ---------------------------------------------------------------------------
# LAYOUT: three columns — Plant Map | Risk Detail | Alert Feed + RAG
# ---------------------------------------------------------------------------
col_map, col_center, col_right = st.columns([1.1, 1.6, 1.3], gap="medium")

# ----------------------- LEFT: M6 Plant heatmap SVG -----------------------
with col_map:
    st.subheader("🗺️ Plant Zone Map")

    # Build ZoneState objects from pipeline state and render M6 SVG
    zone_states_m6 = build_zone_states_from_pipeline(state)
    plant_html = generate_plant_svg(zone_states_m6, pulse_critical=True)
    # Use components.html() so Streamlit doesn't strip <style> CSS animation tags
    components.html(plant_html, height=420, scrolling=False)

    st.caption(
        "Zones colored by compound risk score (0-100). "
        "Orange dashed border = active PTW permit. "
        "Worker dots pulse red when zone is CRITICAL."
    )

    # Score history chart — this is the moment judges watch the number climb
    # live, so it gets real vertical space instead of being a tiny inset.
    if any(state["score_history"].values()):
        st.markdown("**📈 Risk score trend (live)**")
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            for zone_id, history in state["score_history"].items():
                if not history:
                    continue
                ys = [h[1] for h in history]
                color = score_to_tier_color(ys[-1])
                fig.add_trace(go.Scatter(
                    y=ys, mode="lines+markers", name=f"Zone {zone_id}",
                    line=dict(width=3, color=color),
                    marker=dict(size=7),
                ))
            fig.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=30),
                showlegend=True, template="plotly_dark",
                yaxis=dict(
                    range=[0, 100], title="Risk score",
                    gridcolor="rgba(255,255,255,0.12)", showgrid=True,
                    dtick=20,
                ),
                xaxis=dict(
                    title="Evaluation #",
                    gridcolor="rgba(255,255,255,0.06)", showgrid=True,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.caption("Install plotly for the trend chart: pip install plotly")


# ----------------------- CENTER: Current risk + explanation -----------------------
with col_center:
    st.subheader("⚠️ Current Risk Assessment")

    # Find the highest-severity active zone to feature
    severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "SAFE": 0}
    active_zones = [
        (z, zone_severity(z)[0], zone_severity(z)[1])
        for z in ZONE_LAYOUT
        if z in state["risk_events"]
    ]

    if not active_zones:
        st.success("No compound risk conditions detected. All zones nominal.")
    else:
        active_zones.sort(key=lambda x: -severity_rank.get(x[1], 0))
        featured_zone, featured_severity, featured_score = active_zones[0]
        event = state["risk_events"][featured_zone]
        border = SEVERITY_COLORS.get(featured_severity, "#444")

        # Big score display
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px 0;">
                <div style="font-size:64px; font-weight:800; color:{border}; line-height:1;">
                    {featured_score}<span style="font-size:28px; color:#888;">/100</span>
                </div>
                <div style="
                    display:inline-block; background:{border}; color:#fff;
                    padding:4px 18px; border-radius:14px; font-weight:700;
                    letter-spacing:1px; margin-top:8px;
                ">
                    {featured_severity} — ZONE {featured_zone}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Plain-language explanation:**")
        st.markdown(
            f"""<div style="background:#1a1a1a; border-left:4px solid {border};
            padding:14px 16px; border-radius:0 8px 8px 0; font-size:15px; line-height:1.6;">
            {str(event.get('llm_explanation','')).replace(' in None minutes','').replace('None minutes','recently').replace('in None','recently')}
            </div>""",
            unsafe_allow_html=True,
        )

        if event.get("predicted_minutes_to_critical"):
            st.warning(
                f"⏱️ Predicted time to critical: **{event['predicted_minutes_to_critical']} minutes**"
            )

        st.markdown("**🔄 Counterfactual — what would have changed the outcome:**")
        st.markdown(
            f"""<div style="background:#13202e; border:1px solid #2a4a6a;
            padding:14px 16px; border-radius:8px; font-size:14px; line-height:1.6;
            color:#cde4f7;">
            {str(event.get('counterfactual','') or '—').replace('None','—')}
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("")  # spacing

        st.markdown("**Rules fired:**")
        running_total = 0
        arithmetic_parts = []
        for rule in event["rules_fired"]:
            st.markdown(
                f"- `{rule['rule_id']}` **{rule['name']}** (+{rule['score_contribution']} pts)"
            )
            running_total += rule["score_contribution"]
            arithmetic_parts.append(str(rule["score_contribution"]))

        if arithmetic_parts:
            capped_note = " (capped at 100)" if running_total > 100 else ""
            st.caption(
                f"**{' + '.join(arithmetic_parts)} = {min(running_total, 100)}/100{capped_note}** "
                f"— transparent, auditable arithmetic. Not a black box."
            )

        st.markdown("**Recommended actions:**")
        for i, action in enumerate(event.get("recommended_actions", []), 1):
            st.markdown(f"{i}. {action}")

        with st.expander("OISD / regulatory clauses cited"):
            for clause in event.get("oisd_clauses", []):
                st.markdown(f"- {clause}")


# ----------------------- RIGHT: Alert feed + RAG incident memory -----------------------
with col_right:
    st.subheader("📋 Alert Feed")

    if not state["alert_feed"]:
        st.caption("No alerts yet.")
    else:
        for alert in state["alert_feed"][:8]:
            border = SEVERITY_COLORS.get(alert["severity"], "#444")
            ts = alert["evaluated_at"].split("T")[1][:8] if "T" in alert["evaluated_at"] else ""
            st.markdown(
                f"""
                <div style="
                    border-left:3px solid {border}; padding:6px 10px;
                    margin-bottom:8px; background:#161616; border-radius:0 6px 6px 0;
                ">
                    <div style="font-size:12px; color:#888;">{ts} · Zone {alert['zone']}</div>
                    <div style="font-size:13px; color:#ddd;">
                        Score {alert['risk_score']} — {alert['severity']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("📚 Similar Past Incidents")

    if active_zones:
        rag_data = state["rag_context"].get(featured_zone)
        if rag_data:
            st.markdown(f"*{rag_data['headline_match']}*")
            for inc in rag_data["top_incidents"][:3]:
                with st.expander(
                    f"{inc['title']} — {inc['match_percent']}% match ({inc['match_label']})"
                ):
                    st.markdown(f"**Facility:** {inc['facility']}")
                    st.markdown(f"**Fatalities:** {inc['fatalities']}")
                    st.markdown(f"**Summary:** {inc['summary']}")
                    st.markdown("**Root causes:**")
                    for cause in inc["root_causes"]:
                        st.markdown(f"- {cause}")
        else:
            st.caption("Waiting for first risk event to query incident memory...")
    else:
        st.caption("No active risk — no incident lookup triggered.")


# ---------------------------------------------------------------------------
# M5 PERMIT WATCH — active permits vs. live conditions, independent of M3
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📝 Permit Watch (M5)")
st.caption(
    "Every active Permit-to-Work is checked against the exact gas/O₂ thresholds "
    "recorded when it was issued — not generic limits. A permit issued at CO=12ppm "
    "conflicts the moment CO passes the threshold set at that time, independent of "
    "M3's compound score."
)

active_conflicts_all = [
    c for zone_list in state["permit_conflicts"].values() for c in zone_list
]

col_permits, col_conflicts = st.columns([1.4, 1], gap="medium")

with col_permits:
    st.markdown("**Active permits by zone**")
    any_permits = False
    for zone in ZONE_LAYOUT:
        statuses = state["permit_status"].get(zone, [])
        if not statuses:
            continue
        any_permits = True
        for p in statuses:
            sev = p.get("highest_conflict_severity", "NONE")
            badge_color = {
                "CRITICAL": SEVERITY_COLORS["CRITICAL"],
                "HIGH": SEVERITY_COLORS["HIGH"],
                "MEDIUM": SEVERITY_COLORS["MEDIUM"],
                "NONE": SEVERITY_COLORS["SAFE"],
            }.get(sev, "#444")
            expiry_flag = " ⏳ expiring soon" if p.get("expiry_warning") else ""
            st.markdown(
                f"""
                <div style="
                    border-left:3px solid {badge_color}; padding:8px 12px;
                    margin-bottom:8px; background:#161616; border-radius:0 6px 6px 0;
                ">
                    <div style="font-size:13px; color:#eee; font-weight:600;">
                        {p.get('permit_id','')} · Zone {p.get('zone','')} ·
                        {p.get('permit_type','').replace('_',' ').title()}
                    </div>
                    <div style="font-size:12px; color:#999;">
                        {p.get('issued_to','')} — {p.get('work_description','')}
                    </div>
                    <div style="font-size:12px; color:#888;">
                        Valid {p.get('valid_from','')}–{p.get('valid_until','')} ·
                        status: {p.get('status','')}{expiry_flag}
                    </div>
                    <div style="font-size:12px; color:{badge_color}; font-weight:700; margin-top:2px;">
                        {p.get('conflict_count',0)} active conflict(s) — {sev}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if not any_permits:
        st.caption("No active permits loaded for the tracked zones yet.")

with col_conflicts:
    st.markdown("**Live conflict feed**")
    if not state["conflict_feed"]:
        st.caption("No permit conflicts detected yet.")
    else:
        for c in state["conflict_feed"][:8]:
            border = {
                "CRITICAL": SEVERITY_COLORS["CRITICAL"],
                "HIGH": SEVERITY_COLORS["HIGH"],
                "MEDIUM": SEVERITY_COLORS["MEDIUM"],
            }.get(
                "CRITICAL" if c["conflict_type"] in ("OXYGEN_DEPLETION", "HOT_WORK_NO_FIRE_WATCH")
                else "HIGH" if c["conflict_type"] == "GAS_THRESHOLD_BREACH"
                else "MEDIUM",
                "#444",
            )
            ts = c["detected_at"].split("T")[1][:8] if "T" in c["detected_at"] else ""
            with st.expander(
                f"{ts} · {c['permit_id']} · {c['conflict_type'].replace('_',' ').title()}"
            ):
                st.markdown(
                    f"""<div style="border-left:3px solid {border}; padding:8px 12px;
                    background:#161616; border-radius:0 6px 6px 0; font-size:13px; color:#ddd;">
                    {c['description']}
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Threshold `{c['permit_threshold_breached']}` = {c['threshold_value']} · "
                    f"actual = {c['actual_value']} · action: **{c['action_required']}** · "
                    f"{c['oisd_clause']}"
                )
                st.caption(f"Notified: {c.get('notified_contact','—')}")

if active_conflicts_all:
    st.warning(
        f"⚠️ {len(active_conflicts_all)} permit(s) currently in conflict with live "
        f"sensor conditions — see zones above for required action."
    )


# ---------------------------------------------------------------------------
# BOTTOM: Notification log
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📋 Notification Log")

if state["notification_log"]:
    import pandas as pd
    df = pd.DataFrame(state["notification_log"])
    st.dataframe(
        df,
        use_container_width=True,
        height=220,
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
# Auto-refresh loop
# ---------------------------------------------------------------------------
if auto_refresh and st.session_state.pipeline_running:
    time.sleep(2)
    st.rerun()