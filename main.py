#!/usr/bin/env python3
"""
Manufacturing Log Analyzer — CLI

Analyzes manufacturing machine logs using traditional data science methods
combined with LLM-powered root cause analysis.

Usage:
    # Generate sample data
    python main.py generate --num-entries 2000

    # Run full analysis
    python main.py analyze --input data/sample_logs.csv --output reports/

    # Generate data + analyze in one step
    python main.py run --num-entries 2000 --max-windows 5
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

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
from src.analyzer import ManufacturingAnalyzer
from src.report import generate_report, save_report


def cmd_generate(args):
    """Generate synthetic manufacturing logs."""
    logs = generate_logs(
        num_entries=args.num_entries,
        start_date=args.start_date,
        seed=args.seed,
    )
    save_logs(logs, args.output)


def cmd_analyze(args):
    """Run full analysis pipeline on existing logs."""
    print(f"Loading logs from {args.input}...")
    df = load_logs(args.input)
    print(f"  → {len(df)} entries loaded")

    # Traditional analysis
    print("Running feature extraction...")
    df = add_time_features(df)
    df = compute_sensor_stats(df)
    df = flag_anomalies(df)

    severity_summary = get_severity_summary(df)
    print("\nSeverity summary by machine:")
    print(severity_summary.to_string())

    anomaly_count = df["is_anomaly"].sum()
    print(f"\nSensor anomalies detected: {anomaly_count}")

    # Failure window detection
    print("\nIdentifying failure windows...")
    windows = get_failure_windows(df)
    print(f"  → {len(windows)} failure windows found")

    if not windows:
        print("No failure windows detected. Exiting.")
        return

    # Prepare for LLM
    prepared = prepare_for_llm(df, windows, max_windows=args.max_windows)
    print(f"  → {len(prepared)} windows prepared for LLM analysis")

    # LLM analysis
    print(f"\nRunning LLM analysis via {args.provider} (this may take a moment)...")
    analyzer = ManufacturingAnalyzer(
        provider=args.provider,
        api_key=getattr(args, "api_key", None),
    )

    def progress(i, total, machine_id):
        print(f"  Analyzing window {i+1}/{total} — {machine_id}...")

    diagnoses = analyzer.analyze_batch(prepared, progress_callback=progress)

    # Fleet summary
    print("\nGenerating fleet summary...")
    fleet_summary = analyzer.generate_fleet_summary(diagnoses)

    # Generate report
    stats = {
        "Total entries": len(df),
        "Time range": f"{df['timestamp'].min()} → {df['timestamp'].max()}",
        "Machines": ", ".join(sorted(df["machine_id"].unique())),
        "Failure windows": len(windows),
        "Sensor anomalies": anomaly_count,
        "Error rate": f"{(df['severity'].isin(['ERROR', 'CRITICAL']).mean() * 100):.1f}%",
    }

    report = generate_report(diagnoses, fleet_summary, stats)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "diagnostic_report.md"
    save_report(report, report_path)

    # Save processed data
    csv_path = output_dir / "processed_logs.csv"
    df.to_csv(csv_path, index=False)
    print(f"Processed data saved → {csv_path}")

    print(f"\n✅ Analysis complete! Report: {report_path}")


def cmd_run(args):
    """Generate sample data and run full analysis."""
    data_path = Path("data/sample_logs.csv")
    logs = generate_logs(num_entries=args.num_entries, seed=args.seed)
    save_logs(logs, data_path)

    args.input = str(data_path)
    args.output = args.output or "reports"
    cmd_analyze(args)


def main():
    parser = argparse.ArgumentParser(
        description="Manufacturing Log Analyzer — LLM-powered diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic logs")
    gen_parser.add_argument("-n", "--num-entries", type=int, default=2000)
    gen_parser.add_argument("-o", "--output", type=str, default="data/sample_logs.csv")
    gen_parser.add_argument("--start-date", type=str, default="2025-01-01")
    gen_parser.add_argument("--seed", type=int, default=42)

    # Analyze
    ana_parser = subparsers.add_parser("analyze", help="Analyze existing logs")
    ana_parser.add_argument("-i", "--input", type=str, required=True)
    ana_parser.add_argument("-o", "--output", type=str, default="reports")
    ana_parser.add_argument("--max-windows", type=int, default=10)
    ana_parser.add_argument(
        "-p", "--provider", type=str, default="gemini",
        choices=["gemini", "groq", "ollama", "anthropic", "openai"],
        help="LLM provider (default: gemini — free)",
    )
    ana_parser.add_argument("--api-key", type=str, default=None, help="API key (or set env var)")

    # Run (generate + analyze)
    run_parser = subparsers.add_parser("run", help="Generate + analyze in one step")
    run_parser.add_argument("-n", "--num-entries", type=int, default=2000)
    run_parser.add_argument("-o", "--output", type=str, default="reports")
    run_parser.add_argument("--max-windows", type=int, default=5)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "-p", "--provider", type=str, default="gemini",
        choices=["gemini", "groq", "ollama", "anthropic", "openai"],
        help="LLM provider (default: gemini — free)",
    )
    run_parser.add_argument("--api-key", type=str, default=None, help="API key (or set env var)")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
