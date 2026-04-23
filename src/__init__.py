"""Manufacturing Log Analyzer — LLM-powered diagnostics for production environments."""

from .generator import generate_logs, save_logs
from .parser import (
    load_logs,
    add_time_features,
    compute_sensor_stats,
    flag_anomalies,
    get_severity_summary,
    get_failure_windows,
    prepare_for_llm,
)
from .analyzer import ManufacturingAnalyzer, DiagnosisResult, FleetSummary
from .report import generate_report, save_report

__all__ = [
    "generate_logs",
    "save_logs",
    "load_logs",
    "add_time_features",
    "compute_sensor_stats",
    "flag_anomalies",
    "get_severity_summary",
    "get_failure_windows",
    "prepare_for_llm",
    "ManufacturingAnalyzer",
    "DiagnosisResult",
    "FleetSummary",
    "generate_report",
    "save_report",
]
