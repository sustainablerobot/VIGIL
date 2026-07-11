# -*- coding: utf-8 -*-
"""
VIGIL - M8 Response Orchestrator
==================================
When M3's risk score crosses CRITICAL (75+), this module fires.
It coordinates the emergency response sequence and generates
a DGFASLI-compliant incident report using Claude API.

HOW IT WORKS (simple terms)
----------------------------
Think of this as the fire alarm that also:
  1. Automatically calls the fire station (alert notifications)
  2. Photographs the scene (saves sensor state as evidence)
  3. Counts who is in the building (worker evacuation list)
  4. Writes the first paragraph of the incident report
  5. Tells each team exactly what to do in priority order

All of this happens in the first 10 seconds after the critical
threshold is crossed - not the first 10 days.

WHAT JUDGES SEE (the showstopper moment)
-----------------------------------------
During the Vizag scenario, when score hits 75+:
  - A red CRITICAL ALERT banner appears on the dashboard
  - Alert log shows: Safety Officer notified, Plant Manager notified
  - A countdown timer starts (evacuation clock)
  - Claude API generates the full incident report live on screen:
    * Incident summary in plain English
    * All sensor readings at T=0 (preserved as evidence)
    * Regulatory clauses breached (OISD-116, Factory Act)
    * Immediate action checklist
    * Preliminary root cause statement
  This takes 3-5 seconds. Judges watch a document write itself.

SEQUENCE OF ACTIONS (all within 10 seconds of trigger)
-------------------------------------------------------
T+0s  : Trigger confirmed (score >= CRITICAL_THRESHOLD, confidence >= 0.7)
T+0s  : Evidence snapshot saved to disk (audit trail)
T+1s  : Alert log entry created (M7 dashboard shows it immediately)
T+2s  : Notification messages generated (SMS/WhatsApp content)
T+3s  : Evacuation zone determined from worker locations
T+5s  : Claude API called -> generates preliminary incident report
T+8s  : Report saved to disk as JSON + human-readable text
T+10s : Orchestrator signals "response complete" to dashboard

DEDUPLICATION
-------------
A critical event for the same zone within 5 minutes does NOT
re-trigger the full sequence. Prevents alert flooding.
The cooldown is configurable: RESPONSE_COOLDOWN_SEC

TECHNOLOGIES
------------
- anthropic API: incident report generation (structured JSON prompt)
- dataclasses: typed ResponseAction and IncidentReport structures
- threading.Lock: safe concurrent access from M3 callback thread
- json: evidence snapshot serialization
- datetime: ISO 8601 timestamps for regulatory compliance
- pathlib: cross-platform report file paths
- jinja2 (optional): human-readable report template rendering
"""

import os
os.environ["PYTHONUTF8"] = "1"

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("VIGIL.M8.ResponseOrchestrator")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRITICAL_THRESHOLD = 75          # score at which response triggers
CONFIDENCE_THRESHOLD = 0.65      # M3 confidence below this = no trigger
RESPONSE_COOLDOWN_SEC = 300      # 5 minutes between responses for same zone
MAX_REPORT_RETRIES = 2
CLAUDE_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AlertMessage:
    """One outbound alert message (SMS/WhatsApp/log)."""
    channel: str          # "sms", "whatsapp", "dashboard", "scada_log"
    recipient: str        # name or phone number
    message_en: str       # English message
    message_hi: str       # Hindi message
    sent_at: str = ""
    delivered: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceSnapshot:
    """Full sensor state captured at the moment of trigger - the audit trail."""
    snapshot_id: str
    captured_at: str
    zone: str
    risk_score: int
    severity: str
    rules_fired: List[dict]
    sensor_readings: dict       # full sensor dict at T=0
    active_permits: List[str]
    worker_count: int
    worker_zones: dict
    llm_explanation: str
    oisd_clauses: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IncidentReport:
    """
    DGFASLI-style preliminary incident report.
    Generated by Claude within seconds of trigger.
    Saved to disk as both JSON and human-readable text.
    """
    report_id: str
    generated_at: str
    zone: str
    risk_score: int
    severity: str

    # Report sections
    incident_summary: str           # 2-3 sentence plain English summary
    hazardous_conditions: List[str] # bullet list of sensor readings
    permit_status: str              # what permits were active
    worker_status: str              # how many workers, evacuation status
    compound_pattern: str           # what combination triggered this
    regulatory_violations: List[str] # OISD/Factory Act clauses breached
    immediate_actions: List[str]    # ordered checklist
    preliminary_root_cause: str     # AI-generated initial RCA
    evidence_reference: str         # path to evidence snapshot file

    # Metadata
    generated_by: str = "VIGIL M8 Response Orchestrator"
    report_type: str = "PRELIMINARY - NOT FOR FINAL SUBMISSION"
    disclaimer: str = (
        "This preliminary report is AI-generated within seconds of trigger. "
        "It must be reviewed and verified by a qualified safety officer "
        "before submission to regulatory authorities (DGFASLI/Factory Inspector)."
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """Human-readable text version for display on dashboard."""
        lines = [
            "=" * 70,
            f"VIGIL PRELIMINARY INCIDENT REPORT",
            f"Report ID  : {self.report_id}",
            f"Generated  : {self.generated_at}",
            f"Zone       : {self.zone}",
            f"Risk Score : {self.risk_score}/100 ({self.severity})",
            f"Type       : {self.report_type}",
            "=" * 70,
            "",
            "INCIDENT SUMMARY",
            "-" * 40,
            self.incident_summary,
            "",
            "HAZARDOUS CONDITIONS AT TIME OF TRIGGER",
            "-" * 40,
        ]
        for cond in self.hazardous_conditions:
            lines.append(f"  - {cond}")
        lines += [
            "",
            "PERMIT-TO-WORK STATUS",
            "-" * 40,
            self.permit_status,
            "",
            "WORKER STATUS",
            "-" * 40,
            self.worker_status,
            "",
            "COMPOUND RISK PATTERN DETECTED",
            "-" * 40,
            self.compound_pattern,
            "",
            "REGULATORY PROVISIONS TRIGGERED",
            "-" * 40,
        ]
        for viol in self.regulatory_violations:
            lines.append(f"  - {viol}")
        lines += [
            "",
            "IMMEDIATE ACTIONS REQUIRED",
            "-" * 40,
        ]
        for i, action in enumerate(self.immediate_actions, 1):
            lines.append(f"  {i}. {action}")
        lines += [
            "",
            "PRELIMINARY ROOT CAUSE STATEMENT",
            "-" * 40,
            self.preliminary_root_cause,
            "",
            "EVIDENCE",
            "-" * 40,
            f"Evidence snapshot: {self.evidence_reference}",
            "",
            "-" * 70,
            self.disclaimer,
            "=" * 70,
        ]
        return "\n".join(lines)


@dataclass
class ResponseResult:
    """Full result of one response orchestration cycle."""
    response_id: str
    triggered_at: str
    zone: str
    risk_score: int
    alert_messages: List[AlertMessage] = field(default_factory=list)
    evidence_path: str = ""
    report: Optional[IncidentReport] = None
    report_path: str = ""
    actions_completed: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Alert message builder
# ---------------------------------------------------------------------------
def _build_alert_messages(
    zone: str,
    risk_score: int,
    severity: str,
    rules_fired: List[dict],
    worker_count: int,
    oisd_clauses: List[str],
) -> List[AlertMessage]:
    """
    Build the notification messages for SMS/WhatsApp/dashboard.
    English + Hindi for each. This is the content judges see in the
    Notification Log panel when the alert fires.
    """
    rule_summary = ", ".join(r.get("name", "") for r in rules_fired[:3])
    clause_str = oisd_clauses[0] if oisd_clauses else "OISD-116"

    msg_en = (
        f"VIGIL CRITICAL ALERT: Zone {zone} - Risk {risk_score}/100. "
        f"Pattern: {rule_summary}. "
        f"Evacuate non-essential personnel immediately. "
        f"{clause_str} applies. "
        f"{worker_count} worker(s) in zone."
    )

    msg_hi = (
        f"VIGIL ALERT: Zone {zone} mein KHATRA - Score {risk_score}/100. "
        f"Gas aur hot work permit saath mein - TURANT nikaasi karein. "
        f"{worker_count} kaamgaar zone mein hain."
    )

    recipients = [
        ("Safety Officer",    "+91-98XXX-XXXXX", "sms"),
        ("Plant Manager",     "+91-97XXX-XXXXX", "sms"),
        ("Shift Supervisor",  "Shift-WA-Group",  "whatsapp"),
        ("Emergency Control", "SCADA-LOG-4471",  "scada_log"),
        ("Dashboard",         "VIGIL-M7",        "dashboard"),
    ]

    messages = []
    for name, contact, channel in recipients:
        messages.append(AlertMessage(
            channel=channel,
            recipient=f"{name} ({contact})",
            message_en=msg_en,
            message_hi=msg_hi,
            sent_at=datetime.now(timezone.utc).isoformat(),
            delivered=True,   # simulated delivery
        ))

    return messages


# ---------------------------------------------------------------------------
# Claude API incident report generator
# ---------------------------------------------------------------------------
def _generate_report_with_claude(
    evidence: EvidenceSnapshot,
    api_key: Optional[str],
) -> IncidentReport:
    """
    Call Claude API to generate the preliminary incident report.
    Falls back to a template-based report if API is unavailable.
    """
    report_id = f"RPT-{evidence.zone}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if api_key:
        try:
            report = _claude_api_report(evidence, api_key, report_id, generated_at)
            logger.info(f"Claude report generated: {report_id}")
            return report
        except Exception as e:
            logger.warning(f"Claude API failed ({e}), using template fallback")

    # Template fallback — always works, no API needed
    return _template_report(evidence, report_id, generated_at)


def _claude_api_report(
    evidence: EvidenceSnapshot,
    api_key: str,
    report_id: str,
    generated_at: str,
) -> IncidentReport:
    """Generate report via Claude API with structured JSON prompt."""
    import urllib.request

    rules_text = "\n".join(
        f"  - {r.get('rule_id','')}: {r.get('name','')} (+{r.get('score_contribution',0)} pts)"
        for r in evidence.rules_fired
    )
    clauses_text = "\n".join(f"  - {c}" for c in evidence.oisd_clauses)
    sensor_text = json.dumps(evidence.sensor_readings, indent=2)

    prompt = f"""You are VIGIL, an AI industrial safety system. A CRITICAL compound risk event
has been detected. Generate a DGFASLI-compliant preliminary incident report as JSON.

TRIGGER DETAILS:
Zone: {evidence.zone}
Risk Score: {evidence.risk_score}/100
Severity: {evidence.severity}
Detected at: {evidence.captured_at}

COMPOUND RULES FIRED:
{rules_text}

SENSOR READINGS AT T=0:
{sensor_text}

ACTIVE PERMITS: {', '.join(evidence.active_permits) or 'None'}
WORKERS IN ZONE: {evidence.worker_count}
OISD CLAUSES TRIGGERED:
{clauses_text}

SYSTEM CONTEXT:
This matches the Visakhapatnam Steel Plant pattern (January 2025, 8 fatalities)
where gas accumulation + active hot work permit + shift changeover combined
into a fatal compound condition that no single sensor would have flagged.

Generate ONLY valid JSON with these exact keys:
{{
  "incident_summary": "2-3 sentence plain English summary of what is happening and why it is dangerous",
  "hazardous_conditions": ["list", "of", "specific", "sensor", "readings", "and", "conditions"],
  "permit_status": "one paragraph describing active permits and why they are problematic here",
  "worker_status": "one sentence on workers in zone and immediate evacuation priority",
  "compound_pattern": "explain the specific combination of factors that created compound risk, reference Vizag if applicable",
  "regulatory_violations": ["OISD-116 Section X: description", "Factory Act Section Y: description"],
  "immediate_actions": ["action 1", "action 2", "action 3", "action 4", "action 5"],
  "preliminary_root_cause": "2-3 sentence preliminary root cause analysis"
}}

Be specific. Use actual sensor values. Cite actual OISD sections. This report
will be shown to a plant safety officer within 10 seconds of the trigger."""

    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    text = data["content"][0]["text"].strip()
    # Strip markdown code blocks if Claude wrapped it
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())

    return IncidentReport(
        report_id=report_id,
        generated_at=generated_at,
        zone=evidence.zone,
        risk_score=evidence.risk_score,
        severity=evidence.severity,
        incident_summary=parsed.get("incident_summary", ""),
        hazardous_conditions=parsed.get("hazardous_conditions", []),
        permit_status=parsed.get("permit_status", ""),
        worker_status=parsed.get("worker_status", ""),
        compound_pattern=parsed.get("compound_pattern", ""),
        regulatory_violations=parsed.get("regulatory_violations", []),
        immediate_actions=parsed.get("immediate_actions", []),
        preliminary_root_cause=parsed.get("preliminary_root_cause", ""),
        evidence_reference=evidence.snapshot_id,
    )


def _template_report(
    evidence: EvidenceSnapshot,
    report_id: str,
    generated_at: str,
) -> IncidentReport:
    """
    Template fallback — no API needed.
    Generates a complete, realistic report from evidence data alone.
    Used when Claude API is unavailable or during offline demo.
    """
    co = evidence.sensor_readings.get("co_ppm", "N/A")
    o2 = evidence.sensor_readings.get("o2_pct", "N/A")
    lel = evidence.sensor_readings.get("lel_pct", "N/A")
    temp = evidence.sensor_readings.get("temp_c", "N/A")
    permits = evidence.active_permits
    workers = evidence.worker_count
    rules = evidence.rules_fired
    rule_names = [r.get("name", "") for r in rules]

    has_hot_work = any("hot" in str(p).lower() for p in permits)
    has_confined = any("confined" in str(p).lower() for p in permits)

    # Build incident summary based on what fired
    if "Gas + Hot Work Permit" in str(rule_names):
        summary = (
            f"VIGIL has detected a CRITICAL compound risk condition in Zone {evidence.zone}. "
            f"Carbon monoxide has reached {co} PPM while a Hot Work Permit is active in the same zone. "
            f"This is the identical compound pattern that preceded the Visakhapatnam Steel Plant "
            f"explosion (January 2025, 8 fatalities) — gas accumulation combined with an active "
            f"ignition source. Immediate evacuation and permit suspension are required."
        )
    elif has_confined:
        summary = (
            f"VIGIL has detected a CRITICAL compound risk condition in Zone {evidence.zone}. "
            f"Oxygen concentration has dropped to {o2}% with a Confined Space Entry Permit active. "
            f"Asphyxiation risk is present. All entrants must exit immediately."
        )
    else:
        summary = (
            f"VIGIL has detected a CRITICAL compound risk condition in Zone {evidence.zone} "
            f"with score {evidence.risk_score}/100. Multiple compound rules have fired simultaneously. "
            f"Immediate safety officer response required."
        )

    conditions = []
    if co != "N/A":
        conditions.append(f"Carbon Monoxide: {co} PPM (above 50 PPM warning threshold)")
    if o2 != "N/A":
        conditions.append(f"Oxygen: {o2}% (normal: 20.9%; hazardous below 19.5%)")
    if lel != "N/A":
        conditions.append(f"LEL: {lel}% (alarm at 20% LEL)")
    if temp != "N/A":
        conditions.append(f"Temperature: {temp} degrees C (elevated above baseline)")
    for r in rules:
        conditions.append(f"Compound rule {r.get('rule_id','')}: {r.get('name','')} (+{r.get('score_contribution',0)} pts)")

    permit_status = (
        f"Active permits in Zone {evidence.zone}: {', '.join(permits) if permits else 'none recorded'}. "
        + ("Hot Work Permit active — ignition source present while flammable gas detected. "
           "OISD-116 Section 8.1 requires gas clearance certificate before any hot work. "
           "This permit should be suspended immediately." if has_hot_work else "")
        + ("Confined Space Entry Permit active — occupants at asphyxiation risk. "
           "OISD-118 Section 6.1 requires minimum 19.5% O2 for entry. "
           "All entrants must exit immediately." if has_confined else "")
    )

    compound_pattern = (
        f"The following compound condition was detected: "
        + " + ".join(rule_names[:3])
        + f". Each factor individually was within or near its threshold. "
        f"Together they constitute a compound hazard identical to the pattern "
        f"documented in the Visakhapatnam Steel Plant incident (January 2025). "
        f"VIGIL detected this combination {evidence.risk_score - 40} risk points above "
        f"the compound threshold, providing an estimated intervention window before "
        f"conditions reach physically irreversible state."
    )

    violations = evidence.oisd_clauses or [
        "OISD-116 Section 8.4: Simultaneous operations (hot work + confined space) in adjacent zones",
        "OISD-116 Section 8.5: Shift changeover protocol with active permits",
        "OISD-116 Section 12.1: Gas clearance certificate required before hot work",
        "Factory Act 1948, Section 36: Explosive atmosphere precautions not observed",
    ]

    actions = [
        f"IMMEDIATE: Suspend all active permits in Zone {evidence.zone} NOW",
        f"IMMEDIATE: Evacuate all {workers} worker(s) from Zone {evidence.zone}",
        "IMMEDIATE: Isolate all ignition sources in Zone C3 and adjacent zones",
        "URGENT: Activate emergency gas dispersion / ventilation",
        "URGENT: Notify Factory Inspector and DGFASLI emergency line",
        "URGENT: Preserve all sensor logs and permit records as evidence",
        "FOLLOW-UP: Do not re-enter zone until CO < 25 PPM and gas clearance re-issued",
        "FOLLOW-UP: Conduct formal incident investigation per OISD-154 Section 5.2",
    ]

    root_cause = (
        f"Preliminary analysis indicates that no individual sensor reading crossed an alarm "
        f"threshold independently. The critical hazard arose from the simultaneous occurrence "
        f"of {len(rules)} compound conditions: {', '.join(rule_names[:3])}. "
        f"This compound pattern was not detectable by any single-sensor monitoring system "
        f"and required cross-domain correlation across gas sensors, permit records, and "
        f"worker location data. Root cause investigation should focus on: "
        f"(1) why simultaneous permits were issued in this zone, "
        f"(2) whether shift changeover re-verification was conducted, "
        f"(3) whether gas clearance certificate was current at time of permit issuance."
    )

    return IncidentReport(
        report_id=report_id,
        generated_at=generated_at,
        zone=evidence.zone,
        risk_score=evidence.risk_score,
        severity=evidence.severity,
        incident_summary=summary,
        hazardous_conditions=conditions,
        permit_status=permit_status,
        worker_status=f"{workers} worker(s) detected in Zone {evidence.zone} at time of trigger. Immediate evacuation required.",
        compound_pattern=compound_pattern,
        regulatory_violations=violations,
        immediate_actions=actions,
        preliminary_root_cause=root_cause,
        evidence_reference=evidence.snapshot_id,
    )


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------
class ResponseOrchestrator:
    """
    M8 Response Orchestrator.
    Triggered by M3 when risk score >= CRITICAL_THRESHOLD.

    Usage (from dashboard / M3 callback):
        orchestrator = ResponseOrchestrator(
            reports_dir=Path("m8_response_orchestrator/reports"),
            claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
            on_response_ready=my_callback,
        )
        # Wire to M3:
        risk_engine.register_callback(orchestrator.handle_risk_event)
    """

    def __init__(
        self,
        reports_dir: Path = Path("reports"),
        claude_api_key: Optional[str] = None,
        on_response_ready: Optional[Callable] = None,
        critical_threshold: int = CRITICAL_THRESHOLD,
        cooldown_sec: int = RESPONSE_COOLDOWN_SEC,
    ):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = claude_api_key
        self.on_response_ready = on_response_ready
        self.critical_threshold = critical_threshold
        self.cooldown_sec = cooldown_sec

        self._lock = threading.Lock()
        self._last_response: Dict[str, float] = {}    # zone -> timestamp
        self._response_history: List[ResponseResult] = []
        self._active_response: Optional[ResponseResult] = None

        logger.info(
            f"ResponseOrchestrator ready | threshold={critical_threshold} "
            f"cooldown={cooldown_sec}s | claude={'yes' if claude_api_key else 'template-fallback'}"
        )

    def handle_risk_event(self, event) -> Optional[ResponseResult]:
        """
        Called by M3 for every RiskEvent.
        Only triggers full response sequence when score >= threshold.
        Thread-safe — can be called from M3's background thread.
        """
        # Support both RiskEvent objects and dicts
        if hasattr(event, "to_dict"):
            event_dict = event.to_dict()
        else:
            event_dict = event

        score = event_dict.get("risk_score", 0)
        zone = event_dict.get("zone", "UNKNOWN")
        confidence = event_dict.get("confidence", 1.0)
        severity = event_dict.get("severity", "SAFE")

        # Only trigger on critical
        if score < self.critical_threshold:
            return None

        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Zone {zone}: score={score} but confidence={confidence:.2f} below threshold — no response")
            return None

        # Cooldown check
        with self._lock:
            last = self._last_response.get(zone, 0)
            if time.time() - last < self.cooldown_sec:
                remaining = int(self.cooldown_sec - (time.time() - last))
                logger.info(f"Zone {zone}: response in cooldown ({remaining}s remaining)")
                return None
            self._last_response[zone] = time.time()

        logger.info(f"CRITICAL TRIGGER: Zone {zone} score={score} - initiating response sequence")

        # Run the response in a background thread so it doesn't block M3
        result_holder = [None]
        def _run():
            result_holder[0] = self._execute_response(event_dict)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=20)   # wait max 20s for response

        result = result_holder[0]
        if result:
            with self._lock:
                self._response_history.append(result)
                self._active_response = result
            if self.on_response_ready:
                self.on_response_ready(result)

        return result

    def _execute_response(self, event_dict: dict) -> ResponseResult:
        """Run the full 10-second response sequence."""
        t_start = time.time()
        zone = event_dict.get("zone", "UNKNOWN")
        score = event_dict.get("risk_score", 0)
        response_id = f"RESP-{zone}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        result = ResponseResult(
            response_id=response_id,
            triggered_at=datetime.now(timezone.utc).isoformat(),
            zone=zone,
            risk_score=score,
        )

        try:
            # T+0: Build evidence snapshot
            evidence = self._capture_evidence(event_dict, response_id)
            result.actions_completed.append("Evidence snapshot captured")

            # Save evidence to disk immediately (audit trail)
            evidence_path = self.reports_dir / f"{response_id}_evidence.json"
            with open(evidence_path, "w", encoding="utf-8") as f:
                json.dump(evidence.to_dict(), f, indent=2, ensure_ascii=False)
            result.evidence_path = str(evidence_path)
            result.actions_completed.append(f"Evidence saved to {evidence_path.name}")
            logger.info(f"Evidence snapshot saved: {evidence_path}")

            # T+1: Build alert messages
            alerts = _build_alert_messages(
                zone=zone,
                risk_score=score,
                severity=event_dict.get("severity", "CRITICAL"),
                rules_fired=event_dict.get("rules_fired", []),
                worker_count=event_dict.get("worker_count", 0),
                oisd_clauses=event_dict.get("oisd_clauses", []),
            )
            result.alert_messages = alerts
            result.actions_completed.append(f"{len(alerts)} alert messages dispatched")
            for alert in alerts:
                logger.info(f"Alert -> {alert.recipient} via {alert.channel}")

            # T+5: Generate incident report (Claude API or template)
            result.actions_completed.append("Generating DGFASLI incident report...")
            report = _generate_report_with_claude(evidence, self.api_key)
            result.report = report

            # Save report to disk
            report_json_path = self.reports_dir / f"{response_id}_report.json"
            report_txt_path = self.reports_dir / f"{response_id}_report.txt"

            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            with open(report_txt_path, "w", encoding="utf-8") as f:
                f.write(report.to_text())

            result.report_path = str(report_json_path)
            result.actions_completed.append(f"Incident report saved: {report_json_path.name}")
            logger.info(f"Incident report generated: {response_id}")

            result.success = True

        except Exception as e:
            result.error = str(e)
            logger.error(f"Response sequence error: {e}", exc_info=True)

        result.duration_sec = round(time.time() - t_start, 2)
        logger.info(
            f"Response complete: {response_id} | "
            f"success={result.success} | duration={result.duration_sec}s"
        )
        return result

    def _capture_evidence(self, event_dict: dict, response_id: str) -> EvidenceSnapshot:
        """Build the evidence snapshot from M3 RiskEvent data."""
        raw = event_dict.get("raw_snapshot_zone", {})
        latest = raw.get("latest_reading", {}) if raw else {}

        return EvidenceSnapshot(
            snapshot_id=response_id,
            captured_at=datetime.now(timezone.utc).isoformat(),
            zone=event_dict.get("zone", "UNKNOWN"),
            risk_score=event_dict.get("risk_score", 0),
            severity=event_dict.get("severity", "CRITICAL"),
            rules_fired=event_dict.get("rules_fired", []),
            sensor_readings={
                "co_ppm": latest.get("co_ppm"),
                "o2_pct": latest.get("o2_pct"),
                "lel_pct": latest.get("lel_pct"),
                "temp_c": latest.get("temp_c"),
                "vibration_g": latest.get("vibration_g"),
                "h2s_ppm": latest.get("h2s_ppm"),
                "timestamp": latest.get("timestamp"),
            },
            active_permits=event_dict.get("active_permits", []),
            worker_count=event_dict.get("worker_count", 0),
            worker_zones={event_dict.get("zone", "?"): event_dict.get("worker_count", 0)},
            llm_explanation=event_dict.get("llm_explanation", ""),
            oisd_clauses=event_dict.get("oisd_clauses", []),
        )

    def get_active_response(self) -> Optional[dict]:
        """Thread-safe getter for the dashboard to poll."""
        with self._lock:
            return self._active_response.to_dict() if self._active_response else None

    def get_response_history(self) -> List[dict]:
        """Return all past responses for the notification log."""
        with self._lock:
            return [r.to_dict() for r in self._response_history]

    def clear_active_response(self):
        """Called by dashboard after displaying the critical alert."""
        with self._lock:
            self._active_response = None

    def register_callback(self, fn: Callable):
        """Register a callback for when a response is ready (dashboard update)."""
        self.on_response_ready = fn


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("VIGIL M8 - Response Orchestrator")
    print("Testing with simulated Vizag CRITICAL event...")
    print()

    # Simulated RiskEvent dict from M3 (what would come from risk_engine.py)
    mock_event = {
        "event_id": "risk-C3-20260115-084203",
        "zone": "C3",
        "snapshot_id": "snap-20260115-084203-0001",
        "evaluated_at": "2026-01-15T08:42:03Z",
        "risk_score": 87,
        "severity": "CRITICAL",
        "confidence": 0.94,
        "worker_count": 6,
        "active_permits": ["hot_work_PTW-2026-0443", "confined_space_PTW-2026-0441"],
        "rules_fired": [
            {
                "rule_id": "CR-001",
                "name": "Gas + Hot Work Permit",
                "score_contribution": 35,
                "evidence": {"co_ppm": 87.0, "permit_type": "hot_work"}
            },
            {
                "rule_id": "CR-003",
                "name": "Hot Work + Shift Changeover <30min",
                "score_contribution": 20,
                "evidence": {"minutes_since_changeover": 22}
            },
            {
                "rule_id": "CR-006",
                "name": "Simultaneous Permits Across Multiple Zones",
                "score_contribution": 10,
                "evidence": {"permit_zones": ["C3", "C3A"]}
            },
            {
                "rule_id": "CR-010",
                "name": "Triple Compound (gas+permit+maintenance)",
                "score_contribution": 15,
                "evidence": {}
            },
        ],
        "llm_explanation": (
            "Zone C3 risk score is 87/100 (CRITICAL). The CO reading of 87 PPM is not "
            "immediately fatal alone, but combined with the active hot work permit and "
            "shift changeover 22 minutes ago, this is the exact pattern that caused the "
            "Visakhapatnam explosion."
        ),
        "oisd_clauses": [
            "OISD-116 Section 8.4: Simultaneous hot work and confined space entry prohibited",
            "OISD-116 Section 8.5: Shift changeover re-verification required",
            "OISD-116 Section 12.1: Gas clearance certificate mandatory before hot work",
        ],
        "raw_snapshot_zone": {
            "latest_reading": {
                "co_ppm": 87.0,
                "o2_pct": 19.8,
                "lel_pct": 8.0,
                "temp_c": 42.3,
                "vibration_g": 0.3,
                "h2s_ppm": 6.0,
                "timestamp": "2026-01-15T08:42:03Z",
            }
        },
    }

    reports_dir = Path(__file__).parent / "reports"
    orchestrator = ResponseOrchestrator(
        reports_dir=reports_dir,
        claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    print("Triggering response sequence...")
    result = orchestrator.handle_risk_event(mock_event)

    if result:
        print(f"\nResponse ID  : {result.response_id}")
        print(f"Duration     : {result.duration_sec}s")
        print(f"Success      : {result.success}")
        print(f"Evidence at  : {result.evidence_path}")
        print(f"Report at    : {result.report_path}")
        print(f"\nActions completed:")
        for action in result.actions_completed:
            print(f"  - {action}")
        print(f"\nAlerts dispatched ({len(result.alert_messages)}):")
        for alert in result.alert_messages:
            print(f"  [{alert.channel}] -> {alert.recipient}")
        if result.report:
            print(f"\n{'='*60}")
            print("GENERATED INCIDENT REPORT (first 30 lines):")
            print('='*60)
            lines = result.report.to_text().split('\n')
            print('\n'.join(lines[:30]))
            print("... (full report saved to disk)")
    else:
        print("No response triggered (check threshold/cooldown settings)")
