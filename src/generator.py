"""
Synthetic manufacturing log generator.

Generates realistic laser cutting / welding machine logs with:
- Normal operations, warnings, errors, and critical failures
- Sensor telemetry (power, temperature, gas pressure, beam alignment)
- Operator notes (unstructured text — the key challenge for LLM analysis)
- Correlated failure patterns (e.g., temperature drift → beam misalignment → defect)
"""

import csv
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MACHINES = ["LC-01", "LC-02", "LC-03", "WD-01", "WD-02"]
OPERATORS = ["A. Petrov", "M. Chen", "J. Rodriguez", "K. Nakamura", "S. Williams"]
MATERIALS = ["SS304", "SS316", "AL6061", "AL5052", "MILD_STEEL", "TITANIUM_GR5"]
SHIFTS = ["DAY", "SWING", "NIGHT"]

SEVERITY_WEIGHTS = {
    "INFO": 0.55,
    "WARNING": 0.25,
    "ERROR": 0.15,
    "CRITICAL": 0.05,
}

# Normal operating ranges per sensor
SENSOR_RANGES = {
    "laser_power_w": (1800, 2200),
    "temperature_c": (20, 35),
    "gas_pressure_psi": (140, 160),
    "beam_alignment_mm": (0.0, 0.05),
    "feed_rate_mm_min": (1800, 2400),
    "coolant_flow_lpm": (8.0, 12.0),
}

# ---------------------------------------------------------------------------
# Failure scenarios with correlated sensor drift + operator notes
# ---------------------------------------------------------------------------

FAILURE_SCENARIOS = [
    {
        "name": "thermal_runaway",
        "category": "THERMAL",
        "sensor_overrides": {
            "temperature_c": (55, 85),
            "laser_power_w": (2300, 2600),
            "coolant_flow_lpm": (3.0, 5.5),
        },
        "error_codes": ["E-201", "E-202", "E-210"],
        "notes": [
            "Coolant pump making grinding noise, flow rate dropping. Noticing heat marks on housing.",
            "Temp climbing fast. Tried cycling coolant — didn't help. Shutting down for inspection.",
            "Overtemp alarm triggered twice in 20 min. Chiller seems undersized for this job.",
            "Burned smell near coolant reservoir. Possible pump bearing failure.",
            "Parts coming out discolored — heat tint on SS316. Paused production.",
            "Temperature spiked to 78C during thick plate cut. Emergency stop activated.",
        ],
    },
    {
        "name": "beam_misalignment",
        "category": "OPTICAL",
        "sensor_overrides": {
            "beam_alignment_mm": (0.15, 0.45),
            "laser_power_w": (1500, 1750),
        },
        "error_codes": ["E-301", "E-302", "E-315"],
        "notes": [
            "Cut quality degraded — kerf width inconsistent on left side. Suspect mirror drift.",
            "Beam profiler shows asymmetric pattern. Need to recalibrate focusing optics.",
            "Noticing taper on cuts. Checked nozzle — clean. Likely alignment issue.",
            "Replaced nozzle but problem persists. Beam path needs full inspection.",
            "Edge quality failing QC on 40% of parts this batch. Alignment check overdue.",
            "Operator reports visible sparking pattern shifted. Requesting maintenance.",
        ],
    },
    {
        "name": "gas_system_failure",
        "category": "PNEUMATIC",
        "sensor_overrides": {
            "gas_pressure_psi": (80, 110),
            "feed_rate_mm_min": (1000, 1500),
        },
        "error_codes": ["E-401", "E-402", "E-410"],
        "notes": [
            "Assist gas pressure dropping intermittently. Changed regulator — still fluctuating.",
            "Oxide layer forming on cuts — classic low-pressure symptom. Checking supply lines.",
            "Heard hissing near gas manifold. Possible leak at fitting on line 3.",
            "Pressure gauge reads 95 psi but should be 150. Solenoid valve may be stuck.",
            "Switched to backup nitrogen tank. Primary might have bad regulator.",
            "Dross accumulation on underside of parts — insufficient gas flow for this thickness.",
        ],
    },
    {
        "name": "mechanical_wear",
        "category": "MECHANICAL",
        "sensor_overrides": {
            "feed_rate_mm_min": (800, 1400),
            "beam_alignment_mm": (0.08, 0.20),
        },
        "error_codes": ["E-501", "E-502", "E-520"],
        "notes": [
            "X-axis making clicking sound at high traverse speed. Possible linear guide wear.",
            "Positioning accuracy drifting — parts out of tolerance by 0.15mm on repeat cuts.",
            "Backlash noticeable on direction changes. Ball screw might need replacement.",
            "Vibration on Y-axis during rapid moves. Checked belt tension — within spec.",
            "Encoder error on Z-axis. Cleaned read head but intermittent faults continue.",
            "Gantry resonance at feed rates above 2000 mm/min. Limiting speed for now.",
        ],
    },
    {
        "name": "material_defect",
        "category": "MATERIAL",
        "sensor_overrides": {
            "laser_power_w": (2100, 2500),
            "feed_rate_mm_min": (1200, 1600),
        },
        "error_codes": ["E-601", "E-602"],
        "notes": [
            "Sheet from new supplier has inconsistent thickness — ranging 2.8 to 3.4mm on spec 3.0.",
            "Surface contamination on material — oil residue causing flare-ups during cutting.",
            "Rust spots on mild steel batch. Cleaning before cutting but slowing throughput.",
            "Material hardness varies across the sheet. Adaptive power not compensating enough.",
            "Protective film not peeling cleanly, leaving residue in cut zone.",
            "Wrong material loaded — AL5052 instead of AL6061. Caught after 12 parts.",
        ],
    },
    {
        "name": "electrical_fault",
        "category": "ELECTRICAL",
        "sensor_overrides": {
            "laser_power_w": (0, 800),
            "temperature_c": (15, 20),
        },
        "error_codes": ["E-701", "E-702", "E-710"],
        "notes": [
            "Laser source won't fire — power supply fault indicator lit. No output on oscilloscope.",
            "Intermittent power drops during cutting. UPS showing voltage sag events.",
            "Main contactor chattering. Electrical panel needs inspection.",
            "HMI froze during job — had to hard reboot. Lost part program in memory.",
            "Arc flash near junction box B. Isolated circuit. Need electrician ASAP.",
            "Servo drive faulting on axis 2. Error code points to overcurrent condition.",
        ],
    },
]

NORMAL_NOTES = [
    "Routine operation, no issues.",
    "All parameters nominal. Good cut quality.",
    "Shift handoff — machine running well. Nozzle replaced at start of shift.",
    "Completed batch of 200 parts. No rejects.",
    "Ran preventive maintenance checklist. All items green.",
    "New job setup. Test cuts look good.",
    "Calibration check passed — all axes within tolerance.",
    "Steady production. Material feeding cleanly.",
    "Minor adjustment to focus height for new material thickness. Running fine now.",
    "End of shift — machine in standby. Cleaned lens and checked gas levels.",
    "",
    "",
    "",
]

WARNING_NOTES = [
    "Nozzle wear detected — replacement scheduled for next shift.",
    "Lens showing slight contamination. Cleaned and monitoring.",
    "Coolant temperature trending up — 2C above baseline. Watching it.",
    "Feed rate auto-reduced by 5% — adaptive control compensating for material variation.",
    "Gas consumption higher than expected this batch. May need to adjust parameters.",
    "Slight vibration detected at high speed. Within acceptable limits for now.",
    "Filter pressure drop approaching service interval. Ordering replacement.",
]


def _pick_severity() -> str:
    """Weighted random severity level."""
    return random.choices(
        list(SEVERITY_WEIGHTS.keys()),
        weights=list(SEVERITY_WEIGHTS.values()),
    )[0]


def _sensor_reading(name: str, overrides: dict | None = None) -> float:
    """Generate a sensor value, optionally from an override range."""
    if overrides and name in overrides:
        lo, hi = overrides[name]
    else:
        lo, hi = SENSOR_RANGES[name]
    val = random.uniform(lo, hi)
    return round(val, 2) if "alignment" in name or "coolant" in name else round(val, 1)


def _generate_entry(
    timestamp: datetime,
    force_scenario: dict | None = None,
) -> dict:
    """Generate a single log entry."""
    severity = _pick_severity()
    machine = random.choice(MACHINES)
    operator = random.choice(OPERATORS)
    material = random.choice(MATERIALS)
    shift = random.choice(SHIFTS)

    scenario = None
    error_code = ""
    note = ""
    overrides = None

    if force_scenario:
        scenario = force_scenario
    elif severity in ("ERROR", "CRITICAL"):
        scenario = random.choice(FAILURE_SCENARIOS)

    if scenario:
        overrides = scenario["sensor_overrides"]
        error_code = random.choice(scenario["error_codes"])
        note = random.choice(scenario["notes"])
        if severity == "INFO":
            severity = "ERROR"
    elif severity == "WARNING":
        note = random.choice(WARNING_NOTES)
    else:
        note = random.choice(NORMAL_NOTES)

    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "machine_id": machine,
        "operator": operator,
        "shift": shift,
        "material": material,
        "severity": severity,
        "error_code": error_code,
        "failure_category": scenario["category"] if scenario else "",
        "laser_power_w": _sensor_reading("laser_power_w", overrides),
        "temperature_c": _sensor_reading("temperature_c", overrides),
        "gas_pressure_psi": _sensor_reading("gas_pressure_psi", overrides),
        "beam_alignment_mm": _sensor_reading("beam_alignment_mm", overrides),
        "feed_rate_mm_min": _sensor_reading("feed_rate_mm_min", overrides),
        "coolant_flow_lpm": _sensor_reading("coolant_flow_lpm", overrides),
        "operator_notes": note,
    }


def _generate_correlated_burst(
    start: datetime,
) -> list[dict]:
    """
    Generate a correlated failure burst: 3-6 entries from the same scenario
    over 10–60 minutes, simulating a developing problem.
    """
    scenario = random.choice(FAILURE_SCENARIOS)
    count = random.randint(3, 6)
    entries = []
    ts = start
    severities = ["WARNING"] + ["ERROR"] * (count - 2) + ["CRITICAL"]
    for i in range(count):
        entry = _generate_entry(ts, force_scenario=scenario)
        entry["severity"] = severities[min(i, len(severities) - 1)]
        entry["machine_id"] = entries[0]["machine_id"] if entries else entry["machine_id"]
        entries.append(entry)
        ts += timedelta(minutes=random.randint(3, 15))
    return entries


def generate_logs(
    num_entries: int = 2000,
    start_date: str = "2025-01-01",
    burst_probability: float = 0.03,
    seed: int = 42,
) -> list[dict]:
    """
    Generate a full dataset of synthetic manufacturing logs.

    Args:
        num_entries: Approximate number of log entries to generate.
        start_date: Start date for the log timeline.
        burst_probability: Probability of a correlated failure burst at each step.
        seed: Random seed for reproducibility.

    Returns:
        List of log entry dictionaries, sorted by timestamp.
    """
    random.seed(seed)
    entries = []
    ts = datetime.strptime(start_date, "%Y-%m-%d")

    while len(entries) < num_entries:
        if random.random() < burst_probability:
            burst = _generate_correlated_burst(ts)
            entries.extend(burst)
            ts = datetime.strptime(burst[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
        else:
            entries.append(_generate_entry(ts))

        # Advance 5–30 minutes between independent entries
        ts += timedelta(minutes=random.randint(5, 30))

    entries.sort(key=lambda e: e["timestamp"])
    return entries[:num_entries]


def save_logs(entries: list[dict], path: str | Path) -> Path:
    """Write log entries to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(entries[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)
    print(f"Generated {len(entries)} log entries → {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic manufacturing logs")
    parser.add_argument("-n", "--num-entries", type=int, default=2000, help="Number of entries")
    parser.add_argument("-o", "--output", type=str, default="data/sample_logs.csv", help="Output path")
    parser.add_argument("--start-date", type=str, default="2025-01-01", help="Start date")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logs = generate_logs(
        num_entries=args.num_entries,
        start_date=args.start_date,
        seed=args.seed,
    )
    save_logs(logs, args.output)
