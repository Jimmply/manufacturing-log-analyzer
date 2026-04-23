"""
Manufacturing Log Analyzer — Streamlit Dashboard

Interactive web UI for exploring machine logs, viewing sensor trends,
and running LLM-powered failure analysis.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

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
from src.report import generate_report


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Manufacturing Log Analyzer",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Manufacturing Log Analyzer")
st.caption("LLM-powered diagnostics for laser cutting & welding environments")


# ---------------------------------------------------------------------------
# Sidebar — Data source
# ---------------------------------------------------------------------------

st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Generate synthetic data", "Upload CSV"],
)

df = None

if data_source == "Generate synthetic data":
    num_entries = st.sidebar.slider("Number of entries", 500, 5000, 2000, 100)
    seed = st.sidebar.number_input("Random seed", value=42)

    if st.sidebar.button("Generate Data", type="primary"):
        with st.spinner("Generating synthetic logs..."):
            logs = generate_logs(num_entries=num_entries, seed=seed)
            save_logs(logs, "data/sample_logs.csv")
            st.session_state["data_path"] = "data/sample_logs.csv"
            st.session_state["generated"] = True

    if st.session_state.get("generated"):
        df = load_logs(st.session_state["data_path"])

else:
    uploaded = st.sidebar.file_uploader("Upload manufacturing logs (CSV)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

# Load existing data if available
if df is None and Path("data/sample_logs.csv").exists():
    df = load_logs("data/sample_logs.csv")
    st.sidebar.info("Loaded existing data from `data/sample_logs.csv`")

if df is None:
    st.info("👈 Generate or upload data to get started.")
    st.stop()


# ---------------------------------------------------------------------------
# Process data
# ---------------------------------------------------------------------------

df = add_time_features(df)
df = compute_sensor_stats(df)
df = flag_anomalies(df)

# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

tab_overview, tab_sensors, tab_failures, tab_llm = st.tabs(
    ["📊 Overview", "📈 Sensor Trends", "🚨 Failure Windows", "🤖 LLM Analysis"]
)

with tab_overview:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", f"{len(df):,}")
    col2.metric(
        "Error Rate",
        f"{df['severity'].isin(['ERROR', 'CRITICAL']).mean() * 100:.1f}%",
    )
    col3.metric("Machines", df["machine_id"].nunique())
    col4.metric("Anomalies Detected", int(df["is_anomaly"].sum()))

    st.subheader("Events by Severity")
    severity_counts = df["severity"].value_counts().reindex(
        ["INFO", "WARNING", "ERROR", "CRITICAL"], fill_value=0
    )
    fig_sev = px.bar(
        x=severity_counts.index,
        y=severity_counts.values,
        color=severity_counts.index,
        color_discrete_map={
            "INFO": "#4CAF50",
            "WARNING": "#FF9800",
            "ERROR": "#F44336",
            "CRITICAL": "#9C27B0",
        },
        labels={"x": "Severity", "y": "Count"},
    )
    fig_sev.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_sev, use_container_width=True)

    st.subheader("Events by Machine")
    severity_by_machine = get_severity_summary(df)
    fig_machine = px.bar(
        severity_by_machine,
        barmode="stack",
        color_discrete_map={
            "INFO": "#4CAF50",
            "WARNING": "#FF9800",
            "ERROR": "#F44336",
            "CRITICAL": "#9C27B0",
        },
        labels={"value": "Count", "machine_id": "Machine"},
    )
    fig_machine.update_layout(height=350)
    st.plotly_chart(fig_machine, use_container_width=True)

    st.subheader("Failure Categories")
    cat_counts = df[df["failure_category"] != ""]["failure_category"].value_counts()
    if not cat_counts.empty:
        fig_cat = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            hole=0.4,
        )
        fig_cat.update_layout(height=350)
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No categorized failures in dataset.")


# ---------------------------------------------------------------------------
# Sensor trends tab
# ---------------------------------------------------------------------------

with tab_sensors:
    st.header("Sensor Trends")

    sensor_cols = [
        "laser_power_w",
        "temperature_c",
        "gas_pressure_psi",
        "beam_alignment_mm",
        "feed_rate_mm_min",
        "coolant_flow_lpm",
    ]

    selected_machine = st.selectbox(
        "Select machine", sorted(df["machine_id"].unique())
    )
    selected_sensor = st.selectbox("Select sensor", sensor_cols)

    machine_df = df[df["machine_id"] == selected_machine].copy()

    fig_sensor = go.Figure()
    fig_sensor.add_trace(
        go.Scatter(
            x=machine_df["timestamp"],
            y=machine_df[selected_sensor],
            mode="lines+markers",
            name="Actual",
            marker=dict(
                size=4,
                color=[
                    "#F44336" if a else "#2196F3"
                    for a in machine_df.get(f"{selected_sensor}_anomaly", [False] * len(machine_df))
                ],
            ),
            line=dict(color="#2196F3", width=1),
        )
    )

    rolling_col = f"{selected_sensor}_rolling_mean"
    if rolling_col in machine_df.columns:
        fig_sensor.add_trace(
            go.Scatter(
                x=machine_df["timestamp"],
                y=machine_df[rolling_col],
                mode="lines",
                name="Rolling Mean (4h)",
                line=dict(color="#FF9800", width=2, dash="dash"),
            )
        )

    fig_sensor.update_layout(
        title=f"{selected_sensor} — {selected_machine}",
        xaxis_title="Time",
        yaxis_title=selected_sensor,
        height=450,
    )
    st.plotly_chart(fig_sensor, use_container_width=True)

    # Show anomalies
    anomaly_col = f"{selected_sensor}_anomaly"
    if anomaly_col in machine_df.columns:
        anomalies = machine_df[machine_df[anomaly_col]]
        if not anomalies.empty:
            st.warning(f"⚠️ {len(anomalies)} anomalous readings detected for {selected_sensor}")
            st.dataframe(
                anomalies[["timestamp", "severity", selected_sensor, "operator_notes"]].head(20),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Failure windows tab
# ---------------------------------------------------------------------------

with tab_failures:
    st.header("Failure Windows")

    windows = get_failure_windows(df)

    if not windows:
        st.success("No failure windows detected in this dataset.")
    else:
        st.info(f"Found **{len(windows)}** failure windows")

        for i, w in enumerate(windows):
            with st.expander(
                f"🔴 {w['machine_id']} — {w['start'].strftime('%Y-%m-%d %H:%M')} "
                f"({w['event_count']} events, {w['duration_minutes']:.0f} min)"
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Max Severity", w["max_severity"])
                col2.metric("Events", w["event_count"])
                col3.metric("Duration", f"{w['duration_minutes']:.0f} min")

                if w["error_codes"]:
                    st.write(f"**Error codes:** {', '.join(w['error_codes'])}")
                if w["failure_categories"]:
                    st.write(f"**Categories:** {', '.join(w['failure_categories'])}")

                st.write("**Operator Notes:**")
                for note in w["operator_notes"]:
                    if note:
                        st.write(f"- _{note}_")


# ---------------------------------------------------------------------------
# LLM Analysis tab
# ---------------------------------------------------------------------------

with tab_llm:
    st.header("🤖 LLM-Powered Failure Analysis")
    st.write(
        "Uses Claude (Anthropic) to analyze failure windows, classify root causes, "
        "and generate actionable maintenance recommendations."
    )

    api_key = ""
    provider = st.selectbox(
        "LLM Provider",
        ["gemini (free)", "groq (free)", "ollama (free/local)", "anthropic", "openai"],
        help="Gemini and Groq have free tiers. Ollama runs locally with no API key.",
    )
    provider_key = provider.split(" ")[0]

    if provider_key == "ollama":
        st.info("Make sure Ollama is running locally (`ollama serve`) with a model pulled (`ollama pull llama3.1:8b`).")
        api_key = "ollama"
    else:
        env_keys = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
        free_urls = {"gemini": "https://aistudio.google.com/apikey", "groq": "https://console.groq.com/keys"}
        env_key = env_keys.get(provider_key, "")
        help_text = f"Set {env_key} env var to avoid re-entering."
        if provider_key in free_urls:
            help_text += f" Get a free key: {free_urls[provider_key]}"
        api_key = st.text_input(
            f"{provider_key.title()} API Key",
            type="password",
            value=os.environ.get(env_key, ""),
            help=help_text,
        )

    windows = get_failure_windows(df)
    if not windows:
        st.success("No failure windows to analyze.")
        st.stop()

    prepared = prepare_for_llm(df, windows, max_windows=20)

    max_windows = st.slider(
        "Number of failure windows to analyze",
        1,
        min(len(prepared), 20),
        min(5, len(prepared)),
    )

    if st.button("Run LLM Analysis", type="primary", disabled=not api_key):
        try:
            analyzer = ManufacturingAnalyzer(provider=provider_key, api_key=api_key)

            progress_bar = st.progress(0, text="Initializing...")
            diagnoses = []

            for i, window in enumerate(prepared[:max_windows]):
                progress_bar.progress(
                    (i + 1) / (max_windows + 1),
                    text=f"Analyzing {window['machine_id']} ({i+1}/{max_windows})...",
                )
                result = analyzer.analyze_window(window)
                diagnoses.append(result)

            progress_bar.progress(1.0, text="Generating fleet summary...")
            fleet_summary = analyzer.generate_fleet_summary(diagnoses)

            st.session_state["diagnoses"] = diagnoses
            st.session_state["fleet_summary"] = fleet_summary

            progress_bar.empty()
            st.success("✅ Analysis complete!")

        except Exception as e:
            st.error(f"Analysis failed: {e}")

    # Display results
    if "fleet_summary" in st.session_state:
        summary = st.session_state["fleet_summary"]
        st.subheader("Fleet Summary")
        st.info(summary.executive_summary)

        if summary.top_issues:
            st.subheader("Top Issues")
            for issue in summary.top_issues:
                machines = ", ".join(issue.get("affected_machines", []))
                st.write(f"**{issue.get('issue', '')}**")
                st.write(f"- Machines: `{machines}` | Frequency: {issue.get('frequency', 'N/A')}")
                st.write(f"- Impact: {issue.get('business_impact', 'N/A')}")

        if summary.maintenance_priorities:
            st.subheader("Maintenance Priorities")
            mp_df = pd.DataFrame(summary.maintenance_priorities)
            st.dataframe(mp_df, use_container_width=True)

        if summary.trend_alerts:
            st.subheader("Trend Alerts")
            for alert in summary.trend_alerts:
                st.warning(f"⚠️ {alert}")

    if "diagnoses" in st.session_state:
        st.subheader("Detailed Diagnoses")
        for d in st.session_state["diagnoses"]:
            with st.expander(
                f"{'🔴' if d.severity_assessment == 'CRITICAL' else '🟠'} "
                f"{d.machine_id} — {d.failure_mode} ({d.confidence:.0%} confidence)"
            ):
                st.write(f"**Root Cause:** {d.root_cause}")
                st.write(f"**Time:** {d.start} → {d.end}")

                if d.evidence:
                    st.write("**Evidence:**")
                    for ev in d.evidence:
                        st.write(f"- {ev}")

                if d.recommended_actions:
                    st.write("**Recommended Actions:**")
                    for action in d.recommended_actions:
                        st.write(
                            f"- [{action.get('priority', '')}] {action.get('action', '')} "
                            f"(~{action.get('estimated_downtime_hours', '?')}h downtime)"
                        )

                if d.parts_at_risk:
                    st.write(f"**Parts at Risk:** {', '.join(d.parts_at_risk)}")

        # Export report
        st.subheader("Export")
        stats = {
            "Total entries": len(df),
            "Time range": f"{df['timestamp'].min()} → {df['timestamp'].max()}",
            "Machines": ", ".join(sorted(df["machine_id"].unique())),
            "Failure windows analyzed": len(st.session_state["diagnoses"]),
        }
        report_md = generate_report(
            st.session_state["diagnoses"],
            st.session_state.get("fleet_summary"),
            stats,
        )
        st.download_button(
            "📥 Download Full Report (Markdown)",
            data=report_md,
            file_name="diagnostic_report.md",
            mime="text/markdown",
        )
