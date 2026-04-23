"""
Traditional log parsing and feature extraction.

Handles structured fields + extracts statistical features from sensor data.
This is the 'classical' data science layer that feeds into the LLM analyzer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta


def load_logs(path: str | Path) -> pd.DataFrame:
    """Load and parse manufacturing logs from CSV."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    df["time_since_last"] = (
        df.groupby("machine_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )
    return df


def compute_sensor_stats(
    df: pd.DataFrame,
    window_hours: int = 4,
) -> pd.DataFrame:
    """
    Compute rolling statistics for sensor columns per machine.

    Returns the original DataFrame with added rolling mean/std columns.
    """
    sensor_cols = [
        "laser_power_w",
        "temperature_c",
        "gas_pressure_psi",
        "beam_alignment_mm",
        "feed_rate_mm_min",
        "coolant_flow_lpm",
    ]
    df = df.copy()
    df = df.set_index("timestamp").sort_index()

    window = f"{window_hours}h"
    for col in sensor_cols:
        grp = df.groupby("machine_id")[col]
        df[f"{col}_rolling_mean"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_rolling_std"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )

    df = df.reset_index()
    return df


def flag_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Flag sensor readings that deviate significantly from rolling baseline.
    Uses Z-score approach on rolling window statistics.
    """
    sensor_cols = [
        "laser_power_w",
        "temperature_c",
        "gas_pressure_psi",
        "beam_alignment_mm",
        "feed_rate_mm_min",
        "coolant_flow_lpm",
    ]
    df = df.copy()
    anomaly_flags = []

    for col in sensor_cols:
        mean_col = f"{col}_rolling_mean"
        std_col = f"{col}_rolling_std"
        if mean_col in df.columns and std_col in df.columns:
            z_scores = np.where(
                df[std_col] > 0,
                np.abs(df[col] - df[mean_col]) / df[std_col],
                0,
            )
            flag_col = f"{col}_anomaly"
            df[flag_col] = z_scores > z_threshold
            anomaly_flags.append(flag_col)

    df["anomaly_count"] = df[anomaly_flags].sum(axis=1).astype(int)
    df["is_anomaly"] = df["anomaly_count"] > 0
    return df


def get_severity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize event counts by severity and machine."""
    return (
        df.groupby(["machine_id", "severity"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["INFO", "WARNING", "ERROR", "CRITICAL"], fill_value=0)
    )


def get_failure_windows(
    df: pd.DataFrame,
    gap_minutes: int = 60,
) -> list[dict]:
    """
    Identify contiguous failure windows — clusters of ERROR/CRITICAL
    entries on the same machine within `gap_minutes` of each other.

    Returns a list of dicts describing each failure window.
    """
    failures = df[df["severity"].isin(["ERROR", "CRITICAL"])].copy()
    if failures.empty:
        return []

    windows = []
    for machine_id, grp in failures.groupby("machine_id"):
        grp = grp.sort_values("timestamp")
        window_start = None
        window_entries = []

        for _, row in grp.iterrows():
            if window_start is None:
                window_start = row["timestamp"]
                window_entries = [row]
            elif (row["timestamp"] - window_entries[-1]["timestamp"]) <= timedelta(
                minutes=gap_minutes
            ):
                window_entries.append(row)
            else:
                # Close previous window
                windows.append(_summarize_window(machine_id, window_entries))
                window_start = row["timestamp"]
                window_entries = [row]

        if window_entries:
            windows.append(_summarize_window(machine_id, window_entries))

    return sorted(windows, key=lambda w: w["start"])


def _summarize_window(machine_id: str, entries: list) -> dict:
    """Summarize a failure window."""
    timestamps = [e["timestamp"] for e in entries]
    categories = [e.get("failure_category", "") for e in entries if e.get("failure_category")]
    error_codes = [e.get("error_code", "") for e in entries if e.get("error_code")]
    notes = [e.get("operator_notes", "") for e in entries if e.get("operator_notes")]
    severities = [e["severity"] for e in entries]

    return {
        "machine_id": machine_id,
        "start": min(timestamps),
        "end": max(timestamps),
        "duration_minutes": (max(timestamps) - min(timestamps)).total_seconds() / 60,
        "event_count": len(entries),
        "max_severity": "CRITICAL" if "CRITICAL" in severities else "ERROR",
        "failure_categories": list(set(categories)),
        "error_codes": list(set(error_codes)),
        "operator_notes": notes,
        "entries": entries,
    }


def prepare_for_llm(
    df: pd.DataFrame,
    failure_windows: list[dict],
    max_windows: int = 10,
) -> list[dict]:
    """
    Prepare failure windows for LLM analysis.

    Formats each window as a structured context block with sensor data
    and operator notes that the LLM can analyze.
    """
    prepared = []
    for window in failure_windows[:max_windows]:
        entries_data = []
        for entry in window["entries"]:
            entries_data.append({
                "timestamp": str(entry["timestamp"]),
                "severity": entry["severity"],
                "error_code": entry.get("error_code", ""),
                "laser_power_w": entry.get("laser_power_w"),
                "temperature_c": entry.get("temperature_c"),
                "gas_pressure_psi": entry.get("gas_pressure_psi"),
                "beam_alignment_mm": entry.get("beam_alignment_mm"),
                "feed_rate_mm_min": entry.get("feed_rate_mm_min"),
                "coolant_flow_lpm": entry.get("coolant_flow_lpm"),
                "operator_notes": entry.get("operator_notes", ""),
            })

        prepared.append({
            "machine_id": window["machine_id"],
            "start": str(window["start"]),
            "end": str(window["end"]),
            "duration_minutes": round(window["duration_minutes"], 1),
            "event_count": window["event_count"],
            "max_severity": window["max_severity"],
            "error_codes": window["error_codes"],
            "entries": entries_data,
        })

    return prepared
