# -*- coding: utf-8 -*-
"""
VIGIL — M2 Data Fusion Layer
==============================
The central collector. Every 30 seconds (configurable), pulls the latest data
from all four sources and packages them into one unified SensorSnapshot.

Think of it as the secretary that every 30 seconds walks to every department,
collects their latest numbers, and puts everything in one envelope — the
SensorSnapshot — which then gets handed to the Risk Engine (M3).

Without M2, each agent only knows its own slice of data.
M2 gives every downstream module the full picture at the same timestamp.

WHAT THIS MODULE DOES
---------------------
1. Maintains a rolling buffer of the latest sensor reading per zone (from M1)
2. Loads permit records from permits.json (simulates PTW system API)
3. Loads worker locations from workers.json (simulates RFID badge system)
4. Loads maintenance flags from maintenance.json (simulates CMMS system)
5. Every fusion_interval seconds, merges all four into a SensorSnapshot
6. Fires registered callbacks with the snapshot (M3 Risk Engine hooks in here)
7. Detects and flags data staleness — if a sensor hasn't reported in >60s, marks it stale

OUTPUT — SensorSnapshot (every fusion cycle)
--------------------------------------------
{
    "snapshot_id": "snap-20260115-084203",
    "fused_at": "2026-01-15T08:42:03Z",
    "fusion_interval_sec": 30,
    "zones": {
        "C3": {
            "latest_reading": { ...full SensorReading dict... },
            "reading_age_sec": 4.2,           # How old is the sensor data
            "is_stale": false,                 # True if reading > 60s old
            "active_permits": [               # All active permits in this zone
                {
                    "permit_id": "PTW-2026-001",
                    "permit_type": "hot_work",
                    "issued_to": "Ramesh Kumar",
                    "work_description": "Welding on coke oven battery...",
                    "gas_test_result_co_ppm": 12,
                    "fire_watch_assigned": true
                }
            ],
            "workers_present": [              # Workers badged into this zone
                {
                    "worker_id": "W-101",
                    "name": "Ramesh Kumar",
                    "role": "welder",
                    "certified_gas_monitor": true
                }
            ],
            "worker_count": 4,
            "maintenance_active": [           # Equipment under maintenance here
                {
                    "maintenance_id": "MNT-2026-041",
                    "equipment_name": "Coke Oven Battery #3",
                    "maintenance_type": "corrective",
                    "isolation_done": false,
                    "depressurised": false,
                    "notes": "Flange leak repair. Process gas not fully isolated."
                }
            ],
            "has_uncertified_workers": true,  # Any worker without gas monitor cert?
            "has_non_isolated_maintenance": true  # Maintenance without proper isolation?
        }
    },
    "global_flags": {
        "shift_changeover_imminent": true,    # Any zone within 30min of changeover?
        "multi_zone_permit_active": false,    # Permits active in 2+ zones simultaneously?
        "total_workers_on_site": 6,
        "zones_with_active_permits": ["C3", "A1"],
        "zones_with_stale_sensors": []
    },
    "data_sources": {
        "sensor_readings": "live_stream",     # How data arrived
        "permits": "json_file",
        "workers": "json_file",
        "maintenance": "json_file"
    }
}

ALGORITHMS & LOGIC USED
------------------------
1. ROLLING BUFFER (collections.deque per zone)
   M1 fires a reading every second. M2 doesn't process every tick — that would
   overwhelm downstream agents. Instead it keeps only the LATEST reading per zone
   in a dict keyed by zone. Thread-safe via threading.Lock.

2. STALENESS DETECTION
   reading_age_sec = now - reading.timestamp
   is_stale = reading_age_sec > STALE_THRESHOLD_SEC (default: 60)
   A stale sensor is flagged in global_flags.zones_with_stale_sensors.
   This matters because a sensor that stops reporting is itself a safety signal
   (cable damage, power failure) — M3 treats stale sensors as elevated risk.

3. FUSION MERGE STRATEGY
   For each zone that has at least one data source active:
   - Latest sensor reading (from M1 buffer)
   - All permits where permit.zone == zone AND permit.status == "active"
   - All workers where worker.zone == zone AND badge_last_seen within 10 minutes
   - All maintenance where maintenance.zone == zone AND status == "in_progress"
   Cross-reference derived flags are computed here (not in M3) to keep M3 focused on risk.

4. DERIVED SAFETY FLAGS (computed during fusion, not raw data)
   has_uncertified_workers: any(not w.certified_gas_monitor for w in zone_workers)
   has_non_isolated_maintenance: any(not m.isolation_done for m in zone_maintenance)
   shift_changeover_imminent: any zone with shift_changeover_in_min <= 30
   These are pre-computed signals that M3's rule engine can use directly
   without re-implementing the join logic.

5. PERIODIC FUSION TIMER (threading.Timer chain)
   Each fusion cycle schedules the next one using threading.Timer.
   This is a "timer chain" pattern — more reliable than a while+sleep loop
   because it doesn't accumulate drift from fusion processing time itself.

TECHNOLOGIES
------------
- threading.Lock: mutex for thread-safe sensor buffer writes (M1 writes, M2 reads)
- threading.Timer: drift-safe periodic execution
- collections.defaultdict: zone-keyed data grouping
- dataclasses: typed snapshot structure
- json: permit/worker/maintenance file loading
- datetime: UTC timestamp handling and staleness arithmetic
- uuid / hashlib: deterministic snapshot ID generation
"""

import json
import logging
import sys
import threading
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Optional, Any

# Windows PowerShell defaults to a legacy codepage that crashes on em-dashes
# and other non-ASCII characters used in this module's print statements.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "data_fusion.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("VIGIL.M2.DataFusion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STALE_THRESHOLD_SEC = 60          # Sensor reading older than this is flagged stale
WORKER_BADGE_TIMEOUT_SEC = 600    # Worker not seen for 10 min = not in zone
SHIFT_CHANGEOVER_WARN_MIN = 30    # Flag if changeover within this many minutes
DEFAULT_FUSION_INTERVAL = 30      # Seconds between fusion cycles


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ZoneSnapshot:
    """Fused state of a single plant zone at one point in time."""
    zone: str
    latest_reading: Optional[dict]            # Raw SensorReading dict from M1
    reading_age_sec: float                    # Age of sensor reading in seconds
    is_stale: bool                            # True if reading older than STALE_THRESHOLD_SEC
    active_permits: list                      # Active PTW records for this zone
    workers_present: list                     # Workers currently badged in zone
    worker_count: int                         # Derived from workers_present
    maintenance_active: list                  # In-progress maintenance for this zone
    has_uncertified_workers: bool             # Any worker without gas monitor cert
    has_non_isolated_maintenance: bool        # Any maintenance without isolation
    shift_changeover_in_min: Optional[int]    # From latest sensor reading


@dataclass
class GlobalFlags:
    """Plant-wide derived signals computed across all zones."""
    shift_changeover_imminent: bool           # Any zone within SHIFT_CHANGEOVER_WARN_MIN
    multi_zone_permit_active: bool            # Active permits in 2+ zones
    total_workers_on_site: int
    zones_with_active_permits: list
    zones_with_stale_sensors: list
    zones_with_active_maintenance: list


@dataclass
class SensorSnapshot:
    """
    The unified envelope passed to M3 Risk Engine every fusion cycle.
    Contains everything known about the plant at this moment.
    """
    snapshot_id: str
    fused_at: str                             # ISO-8601 UTC
    fusion_interval_sec: int
    zones: dict                               # zone_id -> ZoneSnapshot dict
    global_flags: dict                        # GlobalFlags dict
    data_sources: dict                        # Which sources contributed data

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_zone(self, zone_id: str) -> Optional[dict]:
        return self.zones.get(zone_id)


# ---------------------------------------------------------------------------
# Data loaders — simulate PTW system, RFID badge API, CMMS
# ---------------------------------------------------------------------------
class PermitStore:
    """
    Loads permit records from permits.json.
    In production: replace _load() with an HTTP call to the PTW software API.
    """
    def __init__(self, data_dir: Path):
        self._path = data_dir / "permits.json"
        self._permits: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._permits = data
            logger.info(f"PermitStore: loaded {len(data)} permits from {self._path.name}")
        except FileNotFoundError:
            logger.warning(f"PermitStore: {self._path} not found. Using empty permit list.")
            self._permits = []

    def get_active_for_zone(self, zone: str) -> list[dict]:
        """Return all active permits for a given zone."""
        with self._lock:
            return [
                p for p in self._permits
                if p.get("zone") == zone and p.get("status") == "active"
            ]

    def reload(self) -> None:
        """Hot-reload from disk — call this if permit file changes during operation."""
        self._load()


class WorkerStore:
    """
    Loads worker location data from workers.json.
    In production: replace _load() with RFID badge reader API poll.
    """
    def __init__(self, data_dir: Path):
        self._path = data_dir / "workers.json"
        self._workers: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._workers = data
            logger.info(f"WorkerStore: loaded {len(data)} worker records")
        except FileNotFoundError:
            logger.warning(f"WorkerStore: {self._path} not found. Using empty worker list.")
            self._workers = []

    def get_present_in_zone(self, zone: str, now: datetime) -> list[dict]:
        """
        Return workers currently in zone.
        A worker is 'present' if badge_last_seen within WORKER_BADGE_TIMEOUT_SEC.
        """
        cutoff = now - timedelta(seconds=WORKER_BADGE_TIMEOUT_SEC)
        result = []
        with self._lock:
            for w in self._workers:
                if w.get("zone") != zone:
                    continue
                try:
                    last_seen = datetime.fromisoformat(
                        w["badge_last_seen"].replace("Z", "+00:00")
                    )
                    if last_seen >= cutoff:
                        result.append(w)
                except (KeyError, ValueError):
                    pass
        return result

    def reload(self) -> None:
        self._load()


class MaintenanceStore:
    """
    Loads maintenance records from maintenance.json.
    In production: replace _load() with CMMS (e.g. SAP PM) API call.
    """
    def __init__(self, data_dir: Path):
        self._path = data_dir / "maintenance.json"
        self._records: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._records = data
            logger.info(f"MaintenanceStore: loaded {len(data)} maintenance records")
        except FileNotFoundError:
            logger.warning(f"MaintenanceStore: {self._path} not found.")
            self._records = []

    def get_active_for_zone(self, zone: str) -> list[dict]:
        """Return in-progress maintenance records for a given zone."""
        with self._lock:
            return [
                r for r in self._records
                if r.get("zone") == zone and r.get("status") == "in_progress"
            ]

    def reload(self) -> None:
        self._load()


# ---------------------------------------------------------------------------
# Core DataFusionLayer
# ---------------------------------------------------------------------------
class DataFusionLayer:
    """
    Collects data from all sources and produces a unified SensorSnapshot
    every fusion_interval seconds.

    Usage:
        fusion = DataFusionLayer(fusion_interval=30)
        fusion.register_callback(my_risk_engine.on_snapshot)

        # Wire M1 sensor simulator to M2:
        simulator.register_callback(fusion.ingest_sensor_reading)

        fusion.start()
    """

    def __init__(
        self,
        fusion_interval: int = DEFAULT_FUSION_INTERVAL,
        data_dir: Optional[Path] = None,
        stale_threshold_sec: int = STALE_THRESHOLD_SEC,
    ):
        """
        Args:
            fusion_interval:     Seconds between fusion cycles (default: 30)
            data_dir:            Path to permits.json, workers.json, maintenance.json
            stale_threshold_sec: Max age of sensor reading before flagging stale
        """
        self.fusion_interval = fusion_interval
        self.stale_threshold_sec = stale_threshold_sec
        self.data_dir = data_dir or (Path(__file__).parent / "data")

        # Thread-safe sensor reading buffer: zone -> latest SensorReading dict
        self._sensor_buffer: dict[str, dict] = {}
        self._sensor_timestamps: dict[str, datetime] = {}
        self._buffer_lock = threading.Lock()

        # Data stores
        self._permits = PermitStore(self.data_dir)
        self._workers = WorkerStore(self.data_dir)
        self._maintenance = MaintenanceStore(self.data_dir)

        # Downstream callbacks (M3 Risk Engine hooks in here)
        self._callbacks: list[Callable[[SensorSnapshot], None]] = []

        # Timer chain control
        self._timer: Optional[threading.Timer] = None
        self._running = False

        # Snapshot counter
        self._snapshot_count = 0

        logger.info(
            f"DataFusionLayer initialised | interval={fusion_interval}s | "
            f"stale_threshold={stale_threshold_sec}s | data_dir={self.data_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_callback(self, fn: Callable[[SensorSnapshot], None]) -> None:
        """Register a downstream handler (M3 Risk Engine, dashboard, etc.)"""
        self._callbacks.append(fn)
        logger.info(f"Snapshot callback registered: {fn.__name__}")

    def ingest_sensor_reading(self, reading: Any) -> None:
        """
        Called by M1 SensorSimulator on every tick.
        Accepts either a SensorReading object or a plain dict.
        Stores only the latest reading per zone (overwrites older ones).

        Thread-safe: M1 runs in its own thread, M2 fusion runs in another.
        """
        # Accept both SensorReading objects and raw dicts
        if hasattr(reading, "to_dict"):
            reading_dict = reading.to_dict()
        else:
            reading_dict = dict(reading)

        zone = reading_dict.get("zone", "UNKNOWN")
        now = datetime.now(timezone.utc)

        with self._buffer_lock:
            self._sensor_buffer[zone] = reading_dict
            self._sensor_timestamps[zone] = now

        logger.debug(f"Ingested sensor reading: zone={zone} tick={reading_dict.get('tick')}")

    def start(self) -> None:
        """Start the periodic fusion cycle. Non-blocking — uses timer chain."""
        self._running = True
        logger.info(f"DataFusionLayer started. First fusion in {self.fusion_interval}s.")
        self._schedule_next_fusion()

    def stop(self) -> None:
        """Cancel the next scheduled fusion and stop cleanly."""
        self._running = False
        if self._timer:
            self._timer.cancel()
        logger.info("DataFusionLayer stopped.")

    def fuse_now(self) -> SensorSnapshot:
        """
        Trigger an immediate fusion cycle and return the snapshot.
        Useful for testing and for demo 'trigger now' button.
        """
        return self._fuse()

    def reload_data_sources(self) -> None:
        """Hot-reload all JSON data sources from disk without restarting."""
        self._permits.reload()
        self._workers.reload()
        self._maintenance.reload()
        logger.info("All data sources reloaded from disk.")

    # ------------------------------------------------------------------
    # Internal: timer chain
    # ------------------------------------------------------------------
    def _schedule_next_fusion(self) -> None:
        """Schedule next fusion cycle using threading.Timer chain."""
        if not self._running:
            return
        self._timer = threading.Timer(self.fusion_interval, self._fusion_tick)
        self._timer.daemon = True
        self._timer.name = "FusionTimer"
        self._timer.start()

    def _fusion_tick(self) -> None:
        """Execute one fusion cycle then schedule the next."""
        try:
            snapshot = self._fuse()
            self._fire_callbacks(snapshot)
        except Exception as e:
            logger.error(f"Fusion cycle failed: {e}", exc_info=True)
        finally:
            # Always schedule next cycle even if this one failed
            self._schedule_next_fusion()

    # ------------------------------------------------------------------
    # Internal: core fusion logic
    # ------------------------------------------------------------------
    def _fuse(self) -> SensorSnapshot:
        """
        Merge all four data sources into a SensorSnapshot.

        Steps:
        1. Snapshot the sensor buffer (thread-safe copy)
        2. Determine all known zones (from sensors + permits + maintenance)
        3. For each zone: build ZoneSnapshot with all four data sources
        4. Compute global flags across all zones
        5. Build and return SensorSnapshot
        """
        now = datetime.now(timezone.utc)
        self._snapshot_count += 1

        # Step 1: Thread-safe copy of sensor buffer
        with self._buffer_lock:
            sensor_buffer_copy = dict(self._sensor_buffer)
            sensor_timestamps_copy = dict(self._sensor_timestamps)

        # Step 2: Collect all known zones from all sources
        all_zones = set(sensor_buffer_copy.keys())
        # Also include zones that have permits or maintenance even without sensors
        for permit in self._permits._permits:
            if permit.get("status") == "active":
                all_zones.add(permit["zone"])
        for maint in self._maintenance._records:
            if maint.get("status") == "in_progress":
                all_zones.add(maint["zone"])

        # Step 3: Build per-zone snapshots
        zone_snapshots = {}
        for zone in sorted(all_zones):
            zone_snapshots[zone] = asdict(self._build_zone_snapshot(
                zone, now, sensor_buffer_copy, sensor_timestamps_copy
            ))

        # Step 4: Compute global flags
        global_flags = self._compute_global_flags(zone_snapshots, now)

        # Step 5: Build snapshot ID (deterministic from timestamp + count)
        snap_id = self._make_snapshot_id(now, self._snapshot_count)

        snapshot = SensorSnapshot(
            snapshot_id=snap_id,
            fused_at=now.isoformat(),
            fusion_interval_sec=self.fusion_interval,
            zones=zone_snapshots,
            global_flags=asdict(global_flags),
            data_sources={
                "sensor_readings": "live_stream" if sensor_buffer_copy else "empty",
                "permits": "json_file",
                "workers": "json_file",
                "maintenance": "json_file",
            },
        )

        logger.info(
            f"Fusion #{self._snapshot_count} complete | "
            f"snapshot_id={snap_id} | zones={list(zone_snapshots.keys())} | "
            f"changeover_imminent={global_flags.shift_changeover_imminent} | "
            f"total_workers={global_flags.total_workers_on_site}"
        )

        return snapshot

    def _build_zone_snapshot(
        self,
        zone: str,
        now: datetime,
        sensor_buffer: dict,
        sensor_timestamps: dict,
    ) -> ZoneSnapshot:
        """Build the fused picture for one zone."""

        # --- Sensor data ---
        latest_reading = sensor_buffer.get(zone)
        reading_age_sec = 0.0
        is_stale = True  # Assume stale if no reading

        if latest_reading and zone in sensor_timestamps:
            ts = sensor_timestamps[zone]
            reading_age_sec = (now - ts).total_seconds()
            is_stale = reading_age_sec > self.stale_threshold_sec

        # Shift changeover from sensor reading
        shift_changeover_in_min = None
        if latest_reading:
            shift_changeover_in_min = latest_reading.get("shift_changeover_in_min")

        # --- Permits ---
        active_permits = [
            {
                "permit_id": p["permit_id"],
                "permit_type": p["permit_type"],
                "issued_to": p.get("issued_to", ""),
                "work_description": p.get("work_description", ""),
                "gas_test_result_co_ppm": p.get("gas_test_result_co_ppm"),
                "fire_watch_assigned": p.get("fire_watch_assigned", False),
            }
            for p in self._permits.get_active_for_zone(zone)
        ]

        # --- Workers ---
        raw_workers = self._workers.get_present_in_zone(zone, now)
        workers_present = [
            {
                "worker_id": w["worker_id"],
                "name": w["name"],
                "role": w["role"],
                "certified_gas_monitor": w.get("certified_gas_monitor", False),
            }
            for w in raw_workers
        ]

        # --- Maintenance ---
        raw_maintenance = self._maintenance.get_active_for_zone(zone)
        maintenance_active = [
            {
                "maintenance_id": m["maintenance_id"],
                "equipment_name": m["equipment_name"],
                "maintenance_type": m["maintenance_type"],
                "isolation_done": m.get("isolation_done", False),
                "depressurised": m.get("depressurised", False),
                "notes": m.get("notes", ""),
            }
            for m in raw_maintenance
        ]

        # --- Derived flags ---
        has_uncertified_workers = any(
            not w.get("certified_gas_monitor", False)
            for w in raw_workers
        )
        has_non_isolated_maintenance = any(
            not m.get("isolation_done", False)
            for m in raw_maintenance
        )

        # --- Compute thresholds_breached so M3 _gas_warning() works correctly ---
        # M3 checks latest_reading["thresholds_breached"] to decide if gas is elevated.
        # Without this, all compound rules that depend on gas level never fire.
        GAS_WARN = {"co_ppm": 25.0, "h2s_ppm": 5.0, "ch4_percent_lel": 10.0}
        GAS_ALARM = {"co_ppm": 50.0, "h2s_ppm": 10.0, "ch4_percent_lel": 20.0}
        O2_LOW = 19.5

        if latest_reading:
            breached = []
            breach_levels = {}
            for ch, warn_val in GAS_WARN.items():
                val = latest_reading.get(ch)
                if val is not None:
                    try:
                        fval = float(val)
                        if fval >= GAS_ALARM.get(ch, warn_val * 2):
                            breached.append(ch)
                            breach_levels[ch] = "alarm"
                        elif fval >= warn_val:
                            breached.append(ch)
                            breach_levels[ch] = "warning"
                    except (TypeError, ValueError):
                        pass
            # O2 depletion
            o2 = latest_reading.get("oxygen_percent")
            if o2 is not None:
                try:
                    if float(o2) < O2_LOW:
                        breached.append("oxygen_percent")
                        breach_levels["oxygen_percent"] = "alarm"
                except (TypeError, ValueError):
                    pass
            # Write back into latest_reading so M3 can read it
            latest_reading["thresholds_breached"] = breached
            latest_reading["breach_levels"] = breach_levels

        return ZoneSnapshot(
            zone=zone,
            latest_reading=latest_reading,
            reading_age_sec=round(reading_age_sec, 1),
            is_stale=is_stale,
            active_permits=active_permits,
            workers_present=workers_present,
            worker_count=len(workers_present),
            maintenance_active=maintenance_active,
            has_uncertified_workers=has_uncertified_workers,
            has_non_isolated_maintenance=has_non_isolated_maintenance,
            shift_changeover_in_min=shift_changeover_in_min,
        )

    def _compute_global_flags(self, zone_snapshots: dict, now: datetime) -> GlobalFlags:
        """Compute plant-wide signals from all zone snapshots."""

        zones_with_permits = [
            z for z, snap in zone_snapshots.items()
            if snap.get("active_permits")
        ]
        zones_with_stale = [
            z for z, snap in zone_snapshots.items()
            if snap.get("is_stale") and snap.get("latest_reading") is not None
        ]
        zones_with_maintenance = [
            z for z, snap in zone_snapshots.items()
            if snap.get("maintenance_active")
        ]

        # Shift changeover: any zone within warning window
        changeover_imminent = any(
            snap.get("shift_changeover_in_min") is not None
            and snap["shift_changeover_in_min"] <= SHIFT_CHANGEOVER_WARN_MIN
            for snap in zone_snapshots.values()
        )

        total_workers = sum(
            snap.get("worker_count", 0)
            for snap in zone_snapshots.values()
        )

        return GlobalFlags(
            shift_changeover_imminent=changeover_imminent,
            multi_zone_permit_active=len(zones_with_permits) >= 2,
            total_workers_on_site=total_workers,
            zones_with_active_permits=zones_with_permits,
            zones_with_stale_sensors=zones_with_stale,
            zones_with_active_maintenance=zones_with_maintenance,
        )

    @staticmethod
    def _make_snapshot_id(ts: datetime, count: int) -> str:
        """
        Generate a deterministic, short snapshot ID.
        Format: snap-YYYYMMDD-HHMMSS-NNN
        """
        ts_str = ts.strftime("%Y%m%d-%H%M%S")
        return f"snap-{ts_str}-{count:04d}"

    # ------------------------------------------------------------------
    # Internal: callbacks
    # ------------------------------------------------------------------
    def _fire_callbacks(self, snapshot: SensorSnapshot) -> None:
        for fn in self._callbacks:
            try:
                fn(snapshot)
            except Exception as e:
                logger.error(f"Snapshot callback {fn.__name__} raised: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# CLI entry point — run standalone to see fusion output
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    # Add M1 to path so we can import the simulator
    sys.path.insert(0, str(Path(__file__).parent.parent / "m1_sensor_simulator"))

    try:
        from sensor_simulator import SensorSimulator
        M1_AVAILABLE = True
    except ImportError:
        M1_AVAILABLE = False
        logger.warning("M1 SensorSimulator not found. Running with static test data.")

    def print_snapshot(snapshot: SensorSnapshot) -> None:
        print(f"\n{'='*70}")
        print(f"  VIGIL M2  -  SensorSnapshot  [{snapshot.snapshot_id}]")
        print(f"  Fused at: {snapshot.fused_at}")
        print(f"{'='*70}")

        gf = snapshot.global_flags
        print(f"\n  GLOBAL FLAGS:")
        print(f"    Shift changeover imminent : {gf['shift_changeover_imminent']}")
        print(f"    Multi-zone permits active  : {gf['multi_zone_permit_active']}")
        print(f"    Total workers on site      : {gf['total_workers_on_site']}")
        print(f"    Zones with permits         : {gf['zones_with_active_permits']}")
        print(f"    Zones with stale sensors   : {gf['zones_with_stale_sensors']}")
        print(f"    Zones with maintenance     : {gf['zones_with_active_maintenance']}")

        for zone_id, zone in snapshot.zones.items():
            print(f"\n  ZONE {zone_id} {'(STALE)' if zone['is_stale'] else ''}:")
            if zone["latest_reading"]:
                r = zone["latest_reading"]
                print(f"    Sensor : CO={r.get('co_ppm')}ppm  "
                      f"CH4={r.get('ch4_percent_lel')}%LEL  "
                      f"O2={r.get('oxygen_percent')}%  "
                      f"H2S={r.get('h2s_ppm')}ppm")
                print(f"    Breach : {r.get('thresholds_breached', [])}")
                print(f"    Age    : {zone['reading_age_sec']}s")
            else:
                print(f"    Sensor : No reading yet")

            if zone["active_permits"]:
                for p in zone["active_permits"]:
                    print(f"    Permit : [{p['permit_type'].upper()}] {p['issued_to']}  -  {p['work_description'][:50]}")

            print(f"    Workers: {zone['worker_count']} present | "
                  f"uncertified={zone['has_uncertified_workers']}")

            if zone["maintenance_active"]:
                for m in zone["maintenance_active"]:
                    print(f"    Maint  : {m['equipment_name']} | "
                          f"isolated={m['isolation_done']} "
                          f"depressurised={m['depressurised']}")

    # Build fusion layer with 10s interval for demo (normally 30s)
    fusion = DataFusionLayer(fusion_interval=10)
    fusion.register_callback(print_snapshot)

    if M1_AVAILABLE:
        print("Starting M1 + M2 pipeline (scenario: vizag, fusion every 10s)")
        sim = SensorSimulator(
            scenario="vizag",
            tick_interval=1.0,
            loop=True,
            enable_mqtt=False,
            add_noise=True,
        )
        sim.register_callback(fusion.ingest_sensor_reading)
        fusion.start()

        print("Running. First snapshot in 10s. Press Ctrl+C to stop.\n")
        try:
            sim.start()   # blocking
        except KeyboardInterrupt:
            sim.stop()
            fusion.stop()
    else:
        # No M1 — inject a test reading manually and fuse immediately
        print("Running M2 standalone with injected test reading.\n")
        test_reading = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "zone": "C3",
            "co_ppm": 66.0,
            "h2s_ppm": 6.2,
            "ch4_percent_lel": 16.6,
            "temp_c": 42.0,
            "pressure_bar": 1.10,
            "vibration_g": 0.29,
            "oxygen_percent": 19.9,
            "worker_count": 6,
            "permit_active": True,
            "permit_type": "hot_work",
            "shift_changeover_in_min": 15,
            "scenario": "vizag",
            "tick": 10,
            "thresholds_breached": ["co_ppm", "h2s_ppm", "ch4_percent_lel"],
            "breach_levels": {"co_ppm": "alarm", "h2s_ppm": "alarm", "ch4_percent_lel": "warning"},
            "noise_applied": False,
        }
        fusion.ingest_sensor_reading(test_reading)
        snapshot = fusion.fuse_now()
        print_snapshot(snapshot)
        