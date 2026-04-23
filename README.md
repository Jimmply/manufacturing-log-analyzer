# 🔧 Manufacturing Log Analyzer

**LLM-powered diagnostics for laser cutting & welding environments.**

An end-to-end tool that combines traditional data science (time-series analysis, anomaly detection, feature engineering) with LLM-powered root cause analysis to transform raw machine logs into actionable maintenance recommendations.

Supports multiple LLM providers out of the box — including **free options** (Gemini, Groq, Ollama) — so you can run the full pipeline without spending anything.

Built with real-world manufacturing domain knowledge from analyzing 0.5–1 TB production datasets in laser cutting environments.

---

## What It Does

1. **Generates realistic synthetic manufacturing logs** — sensor telemetry, error codes, operator notes with correlated failure patterns (thermal runaway, beam misalignment, gas system failures, etc.)
2. **Classical analysis pipeline** — rolling statistics, Z-score anomaly detection, failure window clustering, feature extraction
3. **LLM-powered diagnosis** — sends structured failure windows to Claude (Anthropic API) for root cause classification, severity assessment, and maintenance recommendations
4. **Fleet-level reporting** — aggregates findings across machines, identifies cross-machine patterns, prioritizes maintenance actions
5. **Interactive dashboard** — Streamlit UI for exploring sensor trends, failure windows, and LLM analysis results

## Architecture

```
Raw Logs → Parser (Pandas) → Feature Engineering → Anomaly Detection
                                                        ↓
                                              Failure Window Detection
                                                        ↓
                                              LLM Analysis (Claude API)
                                                        ↓
                                        Diagnostic Report + Dashboard
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/Jimmply/manufacturing-log-analyzer.git
cd manufacturing-log-analyzer
pip install -r requirements.txt
```

### 2. Set Up an LLM Provider (pick one — all free options work great)

**Option A: Google Gemini (easiest, free)**
```bash
pip install google-genai
# Get a free key (no credit card): https://aistudio.google.com/apikey
export GEMINI_API_KEY=AIza...
```

**Option B: Groq (free, fastest inference)**
```bash
pip install openai
# Get a free key: https://console.groq.com/keys
export GROQ_API_KEY=gsk_...
```

**Option C: Ollama (fully local, no API key)**
```bash
pip install openai
# Install Ollama: https://ollama.com
ollama pull llama3.1:8b
ollama serve  # keep running in another terminal
```

### 3. Run

**CLI — generate sample data and analyze:**
```bash
# With Gemini (default, free):
python main.py run --num-entries 2000 --max-windows 5

# With Groq (free):
python main.py run -n 2000 --max-windows 5 -p groq

# With Ollama (local):
python main.py run -n 2000 --max-windows 5 -p ollama
```

**CLI — analyze existing logs:**
```bash
python main.py analyze --input your_logs.csv --output reports/ -p gemini
```

**Web dashboard:**
```bash
streamlit run app.py
```

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `generate` | Create synthetic manufacturing logs | `python main.py generate -n 2000 -o data/logs.csv` |
| `analyze` | Run full analysis on existing data | `python main.py analyze -i data/logs.csv -o reports/` |
| `run` | Generate + analyze in one step | `python main.py run -n 2000 --max-windows 5` |

## Input Format

The analyzer expects CSV files with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Event timestamp |
| `machine_id` | string | Machine identifier (e.g., `LC-01`) |
| `severity` | string | `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `error_code` | string | Machine error code (e.g., `E-201`) |
| `laser_power_w` | float | Laser output power in watts |
| `temperature_c` | float | Operating temperature in Celsius |
| `gas_pressure_psi` | float | Assist gas pressure in PSI |
| `beam_alignment_mm` | float | Beam offset from center in mm |
| `feed_rate_mm_min` | float | Cutting feed rate in mm/min |
| `coolant_flow_lpm` | float | Coolant flow rate in liters/min |
| `operator_notes` | string | Free-text operator observations |

## How the LLM Analysis Works

The system doesn't just throw raw logs at an LLM. Instead, it follows a structured pipeline:

1. **Traditional parsing** extracts time features, computes rolling statistics, and flags statistical anomalies using Z-scores
2. **Failure window detection** clusters ERROR/CRITICAL events on the same machine within a configurable time gap
3. **Context preparation** structures each failure window with sensor data, error codes, and operator notes into a formatted prompt
4. **LLM classification** asks Claude to identify root causes, classify failure modes (THERMAL, OPTICAL, PNEUMATIC, MECHANICAL, MATERIAL, ELECTRICAL), assess severity, and recommend specific maintenance actions
5. **Fleet aggregation** sends all individual diagnoses back to Claude for cross-machine pattern detection and maintenance prioritization

This hybrid approach means the LLM focuses on what it does best — interpreting unstructured operator notes and reasoning about complex multi-signal failure patterns — while the classical pipeline handles what it does best — statistical computation and anomaly detection.

## Failure Scenarios

The synthetic data generator models six realistic failure types:

| Scenario | Category | Key Signals |
|----------|----------|-------------|
| Thermal runaway | THERMAL | High temp, low coolant flow, power spikes |
| Beam misalignment | OPTICAL | High alignment offset, power drops |
| Gas system failure | PNEUMATIC | Low gas pressure, reduced feed rate |
| Mechanical wear | MECHANICAL | Feed rate drops, alignment drift |
| Material defect | MATERIAL | Power compensation, feed rate changes |
| Electrical fault | ELECTRICAL | Power loss, temperature drops |

Each scenario includes correlated sensor signatures and realistic operator notes written in natural shop-floor language.

## Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **Python** — Pandas, NumPy for data processing
- **LLM integration** — Provider-agnostic design supporting Gemini (free), Groq (free), Ollama (local), Anthropic, OpenAI
- **Streamlit + Plotly** — interactive dashboard
- **pytest** — testing

## Project Structure

```
manufacturing-log-analyzer/
├── main.py              # CLI entry point
├── app.py               # Streamlit dashboard
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── generator.py     # Synthetic log generation
│   ├── parser.py        # Data parsing + feature engineering
│   ├── analyzer.py      # LLM-powered analysis (Claude API)
│   └── report.py        # Markdown report generation
├── tests/
│   └── test_parser.py   # Unit tests
└── data/                # Generated data (gitignored)
```

## License

MIT
