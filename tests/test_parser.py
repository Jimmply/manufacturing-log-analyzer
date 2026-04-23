"""Tests for the manufacturing log parser module."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import generate_logs, save_logs
from src.parser import (
    load_logs,
    add_time_features,
    compute_sensor_stats,
    flag_anomalies,
    get_severity_summary,
    get_failure_windows,
    prepare_for_llm,
)


@pytest.fixture
def sample_logs(tmp_path):
    """Generate a small reproducible dataset for testing."""
    logs = generate_logs(num_entries=500, seed=42)
    path = tmp_path / "test_logs.csv"
    save_logs(logs, path)
    return path


@pytest.fixture
def df(sample_logs):
    """Load and return a DataFrame from sample logs."""
    return load_logs(sample_logs)


class TestGenerator:
    def test_generates_correct_count(self):
        logs = generate_logs(num_entries=100, seed=1)
        assert len(logs) == 100

    def test_deterministic_with_seed(self):
        logs1 = generate_logs(num_entries=50, seed=99)
        logs2 = generate_logs(num_entries=50, seed=99)
        assert logs1 == logs2

    def test_has_required_fields(self):
        logs = generate_logs(num_entries=10, seed=1)
        required = {
            "timestamp", "machine_id", "operator", "shift",
            "material", "severity", "error_code", "failure_category",
            "laser_power_w", "temperature_c", "gas_pressure_psi",
            "beam_alignment_mm", "feed_rate_mm_min", "coolant_flow_lpm",
            "operator_notes",
        }
        assert required.issubset(set(logs[0].keys()))

    def test_save_and_load(self, tmp_path):
        logs = generate_logs(num_entries=50, seed=1)
        path = save_logs(logs, tmp_path / "test.csv")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 50


class TestParser:
    def test_load_logs(self, df):
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_add_time_features(self, df):
        result = add_time_features(df)
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "time_since_last" in result.columns

    def test_compute_sensor_stats(self, df):
        df = add_time_features(df)
        result = compute_sensor_stats(df)
        assert "laser_power_w_rolling_mean" in result.columns
        assert "temperature_c_rolling_std" in result.columns

    def test_flag_anomalies(self, df):
        df = add_time_features(df)
        df = compute_sensor_stats(df)
        result = flag_anomalies(df)
        assert "is_anomaly" in result.columns
        assert "anomaly_count" in result.columns
        assert result["is_anomaly"].dtype == bool

    def test_severity_summary(self, df):
        summary = get_severity_summary(df)
        assert isinstance(summary, pd.DataFrame)
        assert "INFO" in summary.columns

    def test_failure_windows(self, df):
        windows = get_failure_windows(df)
        assert isinstance(windows, list)
        for w in windows:
            assert "machine_id" in w
            assert "start" in w
            assert "event_count" in w
            assert w["event_count"] > 0

    def test_prepare_for_llm(self, df):
        windows = get_failure_windows(df)
        if windows:
            prepared = prepare_for_llm(df, windows, max_windows=3)
            assert len(prepared) <= 3
            for p in prepared:
                assert "entries" in p
                assert "machine_id" in p
                assert len(p["entries"]) > 0


class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=[
            "timestamp", "machine_id", "severity", "error_code",
            "failure_category", "operator_notes",
            "laser_power_w", "temperature_c", "gas_pressure_psi",
            "beam_alignment_mm", "feed_rate_mm_min", "coolant_flow_lpm",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        windows = get_failure_windows(df)
        assert windows == []

    def test_no_failures(self):
        logs = generate_logs(num_entries=50, seed=1)
        for log in logs:
            log["severity"] = "INFO"
            log["error_code"] = ""
        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        windows = get_failure_windows(df)
        assert windows == []
