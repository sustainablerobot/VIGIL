# -*- coding: utf-8 -*-
"""
VIGIL — M1 Sensor Simulator
============================
Replays scenario CSV files as if real IoT sensors were publishing live.
Publishes to an MQTT broker (or fires callbacks directly if no broker available).

WHAT THIS MODULE DOES
---------------------
1. Loads a scenario CSV (normal / vizag / gas_leak / confined_space)
2. Replays each row at configurable speed (default: 1 row/second)
3. Injects realistic sensor noise so readings never look artificially clean
4. Tracks threshold breaches per sensor channel in real time
5. Publishes each reading as a JSON dict to MQTT topic vigil/sensors/{zone}
6. Fires registered Python callbacks for downstream modules (no MQTT needed in tests)

OUTPUT FORMAT (every tick)
--------------------------
{
    "timestamp": "2026-01-15T08:42:03.112Z",   # ISO-8601 UTC
    "zone": "C3",                                # Plant zone identifier
    "co_ppm": 45.3,                             # Carbon monoxide (ppm)
    "h2s_ppm": 3.1,                             # Hydrogen sulphide (ppm)
    "ch4_percent_lel": 12.7,                    # Methane (% of Lower Explosive Limit)
    "temp_c": 38.4,                             # Temperature (Celsius)
    "pressure_bar": 1.06,                       # Process pressure (bar)
    "vibration_g": 0.21,                        # Equipment vibration (g-force)
    "oxygen_percent": 20.1,                     # O2 level (%)
    "worker_count": 4,                          # Workers in zone (from badge system)
    "permit_active": true,                      # Is any permit currently active?
    "permit_type": "hot_work",                  # Type: hot_work / confined_space / none
    "shift_changeover_in_min": 18,              # Minutes until next shift change
    "scenario": "vizag",                        # Which scenario is running
    "tick": 7,                                  # Row index (0-based)
    "thresholds_breached": ["co_ppm", "ch4_percent_lel"],  # Which channels are above warning
    "noise_applied": true                       # Whether noise was added this tick
}

ALGORITHMS & LOGIC USED
------------------------
1. GAUSSIAN NOISE INJECTION (numpy.random.normal)
   Each sensor channel has a noise_std_dev calibrated to real sensor specs:
   - Gas sensors (CO, H2S, CH4): ±2-5% of reading (electrochemical cell drift)
   - Temperature: ±0.3°C (thermocouple resolution)
   - Pressure: ±0.005 bar (diaphragm sensor accuracy)
   - Vibration: ±0.01g (MEMS accelerometer noise floor)
   Formula: noisy_value = base_value + N(0, noise_std_dev)
   Clipped to physically possible range (no negative gas readings).

2. THRESHOLD BREACH DETECTION
   Hard-coded against OISD-116 / DGFASLI alarm levels:
   - CO: Warning >25ppm, Alarm >50ppm, Emergency >100ppm
   - H2S: Warning >1ppm, Alarm >5ppm, Emergency >10ppm
   - CH4: Warning >10% LEL, Alarm >20% LEL, Emergency >40% LEL
   - O2: Warning <19.5%, Alarm <18%, Emergency <16%
   - Temp: Warning >45°C (above normal process range)
   Breach level (warning/alarm/emergency) is included in output.

3. REPLAY TIMING (time.sleep + drift correction)
   Target: exactly 1 tick per second (configurable).
   Uses monotonic clock with drift correction to prevent cumulative timing error
   over long scenarios. Same technique used in NTP implementations.

4. MQTT PUBLISH (paho-mqtt)
   Topic: vigil/sensors/{zone}  e.g. vigil/sensors/C3
   QoS: 1 (at-least-once delivery)
   Payload: JSON-encoded reading dict
   Falls back to callback-only mode if broker unreachable (for offline dev/demo).

TECHNOLOGIES
------------
- paho-mqtt: lightweight MQTT client (same protocol ESP32 uses — proves I1 claim)
- numpy: gaussian noise generation
- csv + dataclasses: typed scenario loading
- threading: non-blocking publish loop
- logging: structured logs to logs/simulator.log
"""

import csv
import json
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Windows PowerShell defaults to a legacy codepage that crashes on em-dashes
# and other non-ASCII characters used in this module's print statements.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Optional MQTT import — gracefully degrades if paho not installed
# ---------------------------------------------------------------------------
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning("paho-mqtt not installed. Running in callback-only mode.")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "simulator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("VIGIL.M1.SensorSimulator")


# ---------------------------------------------------------------------------
# OISD-116 / DGFASLI threshold definitions
# Source: OISD-116 "Fire Protection Facilities for Petroleum Refineries"
#         DGFASLI "Safety in Petroleum Industry"
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "co_ppm":            {"warning": 25,   "alarm": 50,   "emergency": 100},
    "h2s_ppm":           {"warning": 1,    "alarm": 5,    "emergency": 10},
    "ch4_percent_lel":   {"warning": 10,   "alarm": 20,   "emergency": 40},
    "oxygen_percent":    {"warning": 19.5, "alarm": 18.0, "emergency": 16.0,
                          "invert": True},   # invert=True means LOW is dangerous
    "temp_c":            {"warning": 45,   "alarm": 60,   "emergency": 80},
    "vibration_g":       {"warning": 0.5,  "alarm": 1.0,  "emergency": 2.0},
}

# Noise standard deviations calibrated to real sensor specs
NOISE_STD = {
    "co_ppm":           1.5,    # electrochemical CO sensor: ~±3% FS
    "h2s_ppm":          0.08,   # electrochemical H2S: ~±5% FS
    "ch4_percent_lel":  0.4,    # catalytic bead: ~±2% LEL absolute
    "temp_c":           0.3,    # K-type thermocouple resolution
    "pressure_bar":     0.005,  # diaphragm pressure transmitter
    "vibration_g":      0.01,   # MEMS accelerometer noise floor
    "oxygen_percent":   0.05,   # electrochemical O2 cell
}

# Physical value floors (nothing can go below these)
VALUE_FLOORS = {
    "co_ppm": 0.0,
    "h2s_ppm": 0.0,
    "ch4_percent_lel": 0.0,
    "temp_c": -50.0,
    "pressure_bar": 0.0,
    "vibration_g": 0.0,
    "oxygen_percent": 0.0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ScenarioRow:
    """One row from a scenario CSV — raw values before noise."""
    timestamp_offset_sec: int
    zone: str
    co_ppm: float
    h2s_ppm: float
    ch4_percent_lel: float
    temp_c: float
    pressure_bar: float
    vibration_g: float
    oxygen_percent: float
    worker_count: int
    permit_active: bool
    permit_type: str
    shift_changeover_in_min: int


@dataclass
class SensorReading:
    """
    Final output dict published per tick.
    This is what all downstream modules (M2 risk engine, M3 RAG, M4 permit agent) consume.
    """
    timestamp: str
    zone: str
    co_ppm: float
    h2s_ppm: float
    ch4_percent_lel: float
    temp_c: float
    pressure_bar: float
    vibration_g: float
    oxygen_percent: float
    worker_count: int
    permit_active: bool
    permit_type: str
    shift_changeover_in_min: int
    scenario: str
    tick: int
    thresholds_breached: list = field(default_factory=list)
    breach_levels: dict = field(default_factory=dict)
    noise_applied: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core simulator class
# ---------------------------------------------------------------------------
class SensorSimulator:
    """
    Replays scenario CSV rows as live sensor readings.

    Usage:
        sim = SensorSimulator(scenario="vizag", tick_interval=1.0)
        sim.register_callback(my_handler)   # downstream modules hook in here
        sim.start()                         # blocking
        # or
        sim.start_async()                   # non-blocking thread
    """

    SCENARIO_FILES = {
        "normal":          "scenario_normal.csv",
        "vizag":           "scenario_vizag.csv",
        "gas_leak":        "scenario_gas_leak.csv",
        "confined_space":  "scenario_confined_space.csv",
        "multizone":       "scenario_multizone.csv",
    }

    def __init__(
        self,
        scenario: str = "vizag",
        tick_interval: float = 1.0,
        loop: bool = True,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        enable_mqtt: bool = True,
        add_noise: bool = True,
        data_dir: Optional[Path] = None,
    ):
        """
        Args:
            scenario:       Which scenario CSV to replay (normal/vizag/gas_leak/confined_space)
            tick_interval:  Seconds between ticks (1.0 = real time, 0.1 = 10x speed for demos)
            loop:           Whether to loop the scenario after the last row
            mqtt_broker:    MQTT broker hostname
            mqtt_port:      MQTT broker port (default 1883)
            enable_mqtt:    Set False to skip MQTT entirely (callback-only mode)
            add_noise:      Whether to apply Gaussian noise (set False for unit tests)
            data_dir:       Override path to CSV files (defaults to ./data/)
        """
        if scenario not in self.SCENARIO_FILES:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Choose from: {list(self.SCENARIO_FILES.keys())}"
            )

        self.scenario = scenario
        self.tick_interval = tick_interval
        self.loop = loop
        self.add_noise = add_noise
        self.data_dir = data_dir or (Path(__file__).parent / "data")

        self._callbacks: list[Callable[[SensorReading], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._rows: list[ScenarioRow] = []

        # MQTT setup
        self._mqtt_client: Optional[object] = None
        if enable_mqtt and MQTT_AVAILABLE:
            self._setup_mqtt(mqtt_broker, mqtt_port)

        # Load scenario
        self._load_scenario()
        logger.info(
            f"SensorSimulator initialised | scenario={scenario} | "
            f"rows={len(self._rows)} | tick={tick_interval}s | "
            f"noise={add_noise} | mqtt={'enabled' if self._mqtt_client else 'disabled'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_callback(self, fn: Callable[[SensorReading], None]) -> None:
        """
        Register a function to be called on every tick with the SensorReading.
        This is how downstream modules (risk engine, permit agent, etc.) consume data.
        """
        self._callbacks.append(fn)
        logger.info(f"Callback registered: {fn.__name__}")

    def start(self) -> None:
        """Start the replay loop — BLOCKING. Use start_async() for background operation."""
        self._running = True
        logger.info(f"Starting sensor replay: scenario='{self.scenario}'")
        self._replay_loop()

    def start_async(self) -> threading.Thread:
        """Start in background thread. Returns the thread object."""
        self._thread = threading.Thread(target=self.start, daemon=True, name="SensorSimulator")
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Signal the replay loop to stop after current tick."""
        self._running = False
        logger.info("SensorSimulator stop requested.")
        if self._mqtt_client:
            self._mqtt_client.disconnect()

    # ------------------------------------------------------------------
    # Internal: scenario loading
    # ------------------------------------------------------------------
    def _load_scenario(self) -> None:
        """Parse scenario CSV into typed ScenarioRow list."""
        csv_path = self.data_dir / self.SCENARIO_FILES[self.scenario]
        if not csv_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {csv_path}")

        self._rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                self._rows.append(ScenarioRow(
                    timestamp_offset_sec=int(raw["timestamp_offset_sec"]),
                    zone=raw["zone"].strip(),
                    co_ppm=float(raw["co_ppm"]),
                    h2s_ppm=float(raw["h2s_ppm"]),
                    ch4_percent_lel=float(raw["ch4_percent_lel"]),
                    temp_c=float(raw["temp_c"]),
                    pressure_bar=float(raw["pressure_bar"]),
                    vibration_g=float(raw["vibration_g"]),
                    oxygen_percent=float(raw["oxygen_percent"]),
                    worker_count=int(raw["worker_count"]),
                    permit_active=raw["permit_active"].strip().lower() == "true",
                    permit_type=raw["permit_type"].strip(),
                    shift_changeover_in_min=int(raw["shift_changeover_in_min"]),
                ))
        logger.info(f"Loaded {len(self._rows)} rows from {csv_path.name}")

    # ------------------------------------------------------------------
    # Internal: replay loop with drift-corrected timing
    # ------------------------------------------------------------------
    def _replay_loop(self) -> None:
        """
        Main loop. Replays rows with monotonic-clock drift correction.

        Drift correction algorithm:
            next_tick_at = start_time + (tick_index + 1) * tick_interval
            sleep_duration = next_tick_at - time.monotonic()
        This prevents cumulative timing drift that would occur with naive time.sleep(interval).
        Same technique used in NTP and audio buffer scheduling.
        """
        start_time = time.monotonic()
        tick = 0
        total_rows = len(self._rows)

        while self._running:
            row_index = tick % total_rows
            row = self._rows[row_index]

            # Build reading with optional noise
            reading = self._build_reading(row, tick)

            # Detect threshold breaches
            self._detect_breaches(reading)

            # Publish via MQTT
            self._mqtt_publish(reading)

            # Fire all registered callbacks
            self._fire_callbacks(reading)

            # Log summary
            breach_str = ", ".join(reading.thresholds_breached) or "none"
            logger.info(
                f"tick={tick:04d} zone={reading.zone} "
                f"CO={reading.co_ppm:.1f}ppm CH4={reading.ch4_percent_lel:.1f}%LEL "
                f"O2={reading.oxygen_percent:.1f}% permit={reading.permit_type} "
                f"changeover_in={reading.shift_changeover_in_min}min "
                f"breached=[{breach_str}]"
            )

            tick += 1

            # Stop if not looping and we've exhausted all rows
            if not self.loop and tick >= total_rows:
                logger.info("Scenario complete. Stopping (loop=False).")
                self._running = False
                break

            # Drift-corrected sleep
            next_tick_at = start_time + tick * self.tick_interval
            sleep_for = next_tick_at - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Internal: reading construction
    # ------------------------------------------------------------------
    def _build_reading(self, row: ScenarioRow, tick: int) -> SensorReading:
        """
        Convert a ScenarioRow into a SensorReading, applying Gaussian noise.

        Noise model:
            For each numeric channel, sample from N(0, std_dev) and add to base.
            Clip to physical floor (no negative gas concentrations).
            Round to 1 decimal place (matches real sensor display resolution).
        """
        rng = np.random.default_rng(seed=None)  # unseeded = fresh randomness each run

        def noisy(value: float, channel: str) -> float:
            if not self.add_noise or channel not in NOISE_STD:
                return round(value, 2)
            noise = rng.normal(0, NOISE_STD[channel])
            result = value + noise
            floor = VALUE_FLOORS.get(channel, 0.0)
            return round(max(floor, result), 2)

        return SensorReading(
            timestamp=datetime.now(timezone.utc).isoformat(),
            zone=row.zone,
            co_ppm=noisy(row.co_ppm, "co_ppm"),
            h2s_ppm=noisy(row.h2s_ppm, "h2s_ppm"),
            ch4_percent_lel=noisy(row.ch4_percent_lel, "ch4_percent_lel"),
            temp_c=noisy(row.temp_c, "temp_c"),
            pressure_bar=noisy(row.pressure_bar, "pressure_bar"),
            vibration_g=noisy(row.vibration_g, "vibration_g"),
            oxygen_percent=noisy(row.oxygen_percent, "oxygen_percent"),
            worker_count=row.worker_count,
            permit_active=row.permit_active,
            permit_type=row.permit_type,
            shift_changeover_in_min=row.shift_changeover_in_min,
            scenario=self.scenario,
            tick=tick,
            noise_applied=self.add_noise,
        )

    # ------------------------------------------------------------------
    # Internal: threshold breach detection
    # ------------------------------------------------------------------
    def _detect_breaches(self, reading: SensorReading) -> None:
        """
        Check each sensor channel against OISD-116 alarm levels.
        Populates reading.thresholds_breached and reading.breach_levels in-place.

        For inverted channels (oxygen — low is dangerous), comparison is flipped.
        Breach severity: warning < alarm < emergency (highest wins per channel).
        """
        breached = []
        levels = {}

        channel_values = {
            "co_ppm":          reading.co_ppm,
            "h2s_ppm":         reading.h2s_ppm,
            "ch4_percent_lel": reading.ch4_percent_lel,
            "oxygen_percent":  reading.oxygen_percent,
            "temp_c":          reading.temp_c,
            "vibration_g":     reading.vibration_g,
        }

        for channel, value in channel_values.items():
            spec = THRESHOLDS.get(channel)
            if not spec:
                continue

            invert = spec.get("invert", False)
            level = None

            if invert:
                # Low value = danger (oxygen depletion)
                if value <= spec["emergency"]:
                    level = "emergency"
                elif value <= spec["alarm"]:
                    level = "alarm"
                elif value <= spec["warning"]:
                    level = "warning"
            else:
                # High value = danger (gas accumulation, heat, vibration)
                if value >= spec["emergency"]:
                    level = "emergency"
                elif value >= spec["alarm"]:
                    level = "alarm"
                elif value >= spec["warning"]:
                    level = "warning"

            if level:
                breached.append(channel)
                levels[channel] = level

        reading.thresholds_breached = breached
        reading.breach_levels = levels

    # ------------------------------------------------------------------
    # Internal: MQTT
    # ------------------------------------------------------------------
    def _setup_mqtt(self, broker: str, port: int) -> None:
        """Connect to MQTT broker. Silently disables if broker unreachable."""
        try:
            client = mqtt.Client(client_id="vigil-sensor-simulator", protocol=mqtt.MQTTv5)
            client.on_connect = lambda c, u, f, rc, p: logger.info(
                f"MQTT connected to {broker}:{port} (rc={rc})"
            )
            client.on_disconnect = lambda c, u, rc: logger.warning(f"MQTT disconnected (rc={rc})")
            client.connect(broker, port, keepalive=60)
            client.loop_start()
            self._mqtt_client = client
            logger.info(f"MQTT client ready: broker={broker}:{port}")
        except Exception as e:
            logger.warning(f"MQTT broker unavailable ({e}). Running callback-only mode.")
            self._mqtt_client = None

    def _mqtt_publish(self, reading: SensorReading) -> None:
        """Publish reading JSON to vigil/sensors/{zone}."""
        if not self._mqtt_client:
            return
        topic = f"vigil/sensors/{reading.zone}"
        payload = reading.to_json()
        try:
            self._mqtt_client.publish(topic, payload, qos=1)
        except Exception as e:
            logger.error(f"MQTT publish failed: {e}")

    # ------------------------------------------------------------------
    # Internal: callbacks
    # ------------------------------------------------------------------
    def _fire_callbacks(self, reading: SensorReading) -> None:
        """Call every registered downstream handler with the reading."""
        for fn in self._callbacks:
            try:
                fn(reading)
            except Exception as e:
                logger.error(f"Callback {fn.__name__} raised: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# CLI entry point — run directly to see readings in terminal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VIGIL M1 Sensor Simulator")
    parser.add_argument(
        "--scenario",
        choices=["normal", "vizag", "gas_leak", "confined_space"],
        default="vizag",
        help="Which scenario to replay (default: vizag)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Tick interval in seconds (0.5 = 2x speed, default: 1.0)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Stop after one pass through the scenario",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable Gaussian noise (clean CSV values only)",
    )
    parser.add_argument(
        "--no-mqtt",
        action="store_true",
        help="Disable MQTT publishing (callback-only mode)",
    )
    args = parser.parse_args()

    # Simple console callback — prints each reading as a formatted summary
    def console_display(reading: SensorReading) -> None:
        breaches = reading.thresholds_breached
        breach_display = ""
        if breaches:
            levels = reading.breach_levels
            parts = [f"{ch.upper()}={levels[ch].upper()}" for ch in breaches]
            breach_display = f"  ⚠ BREACH: {', '.join(parts)}"

        permit_display = f"[PERMIT:{reading.permit_type.upper()}]" if reading.permit_active else ""
        changeover = f"[CHANGEOVER IN {reading.shift_changeover_in_min}min]"

        print(
            f"\n[{reading.timestamp}] Zone {reading.zone} | Tick {reading.tick:03d}"
            f"\n  CO={reading.co_ppm}ppm  H2S={reading.h2s_ppm}ppm  "
            f"CH4={reading.ch4_percent_lel}%LEL  O2={reading.oxygen_percent}%"
            f"\n  Temp={reading.temp_c}°C  Press={reading.pressure_bar}bar  "
            f"Vib={reading.vibration_g}g  Workers={reading.worker_count}"
            f"\n  {permit_display} {changeover}"
            f"{breach_display}"
        )

    sim = SensorSimulator(
        scenario=args.scenario,
        tick_interval=args.speed,
        loop=not args.no_loop,
        enable_mqtt=not args.no_mqtt,
        add_noise=not args.no_noise,
    )
    sim.register_callback(console_display)

    print(f"\n{'='*60}")
    print(f"  VIGIL M1  -  Sensor Simulator")
    print(f"  Scenario : {args.scenario.upper()}")
    print(f"  Speed    : {args.speed}s/tick")
    print(f"  Loop     : {not args.no_loop}")
    print(f"  Noise    : {not args.no_noise}")
    print(f"{'='*60}\n")
    print("Press Ctrl+C to stop.\n")

    try:
        sim.start()
    except KeyboardInterrupt:
        sim.stop()
        print("\nSimulator stopped.")
        