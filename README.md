# VIGIL - Compound Risk Intelligence for Zero-Harm Operations

AI-powered industrial safety monitoring prototype built for ET AI Hackathon 2026.

VIGIL watches multiple sensor and permit data streams from a plant at the same time and looks for dangerous *combinations* of conditions, not just single readings crossing a threshold. Many of the worst industrial accidents happen when several individually-normal signals overlap into a fatal condition that no single sensor would have flagged. VIGIL is built to catch that pattern — see the documentation (`VIGIL_Documentation_updated.pdf`) for the real incidents that motivate this and important caveats about how the demo scenarios relate to them.

## What it does

- Simulates live sensor data (gas, temperature, pressure, O2, etc.) across plant zones
- Fuses sensor readings with worker location, permit, and maintenance data
- Runs 10 compound risk rules (CR-001 to CR-010) against the fused data
- Uses a RAG pipeline (FAISS + LangChain) to pull up similar past incidents
  and relevant regulations (OISD, DGMS, Factory Act) when a risk fires
- Cross-checks active work permits against the gas readings that were recorded when the permit was issued
- Shows everything on a live Streamlit dashboard, with a plant heatmap and an alert feed
- Runs an independent CCTV vision check (M9) for worker presence and PPE compliance — reports to its own dashboard tab, not yet wired into the compound risk score (see Future Scope)
- Generates a DGFASLI-style incident report and WhatsApp/SMS-formatted alert content when a zone crosses a critical risk score (message dispatch itself isn't wired up yet)

## Project structure

```
m1_sensor_simulator/     # replays sensor data from CSV scenarios
m2_data_fusion/          # merges sensor + worker + permit + maintenance data
m3_risk_engine/          # compound risk rules + Claude-based reasoning
m4_rag_incident_memory/  # FAISS incident + regulation lookup
m5_permit_watch/         # permit-vs-sensor conflict checks
m6_plant_heatmap/        # SVG zone map generation
m7_dashboard_ui/         # Streamlit dashboard
m8_response_orchestrator/# alert content + incident report generation
m9_cctv_vision/          # OpenCV (default) / optional YOLOv8 PPE + worker-presence detection
```

Each module runs on its own daemon thread and communicates through callback registration rather than direct calls, so each one can be tested, replaced, or upgraded on its own. M5 (Permit Watch) and M9 (CCTV Vision) currently run independently — their output shows up in their own dashboard tabs but isn't yet wired into M3's risk score (see Future Scope in the documentation).

## Running it

```bash
pip install -r requirements.txt

# add your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

cd m7_dashboard_ui
streamlit run dashboard.py
```

Pick a scenario from the sidebar dropdown and hit Start. `multizone` is the main demo - three zones running at once, one of them escalates to CRITICAL around T=70s.

Available scenarios: `vizag`, `multizone`, `confined_space`, `gas_leak`,`normal`. (`vizag` is a compound-risk scenario inspired by the general pattern behind real gas-leak incidents — not a literal reconstruction of a specific event's facts.)

## Tech used

Python, Streamlit, Claude API, FAISS, sentence-transformers, LangChain, threading, Plotly, pandas, OpenCV (default for M9) / ultralytics YOLOv8 (optional upgrade for M9).

## Status

This is a hackathon prototype, not a production system. Sensor input is currently simulated from CSV files - in a real deployment M1 would be replaced by actual ESP32/MQTT sensors, and the rest of the pipeline stays the same.

## Architecture
![Architecture](Architecture_diagram.png)

## Author

Pratyaksha Gupta - B.Tech IT, AKGEC Ghaziabad (2023-2027)
