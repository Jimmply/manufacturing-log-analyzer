"""
Diagnostic report generator.

Takes LLM analysis results and produces formatted Markdown reports
suitable for engineering teams and operations managers.
"""

from datetime import datetime
from pathlib import Path

from .analyzer import DiagnosisResult, FleetSummary


def _severity_badge(severity: str) -> str:
    """Return a text badge for severity level."""
    badges = {
        "CRITICAL": "🔴 CRITICAL",
        "HIGH": "🟠 HIGH",
        "MEDIUM": "🟡 MEDIUM",
        "LOW": "🟢 LOW",
    }
    return badges.get(severity, f"⚪ {severity}")


def _priority_badge(priority: str) -> str:
    badges = {
        "IMMEDIATE": "⚡ IMMEDIATE",
        "NEXT_SHIFT": "📋 NEXT SHIFT",
        "SCHEDULED": "📅 SCHEDULED",
        "THIS_WEEK": "📋 THIS WEEK",
        "THIS_MONTH": "📅 THIS MONTH",
    }
    return badges.get(priority, priority)


def generate_diagnosis_section(diagnosis: DiagnosisResult) -> str:
    """Generate a report section for a single diagnosis."""
    lines = []
    lines.append(f"### Machine `{diagnosis.machine_id}` — {diagnosis.start} → {diagnosis.end}")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Root Cause** | {diagnosis.root_cause} |")
    lines.append(f"| **Failure Mode** | `{diagnosis.failure_mode}` |")
    lines.append(f"| **Confidence** | {diagnosis.confidence:.0%} |")
    lines.append(f"| **Severity** | {_severity_badge(diagnosis.severity_assessment)} |")
    lines.append("")

    if diagnosis.evidence:
        lines.append("**Evidence:**")
        for ev in diagnosis.evidence:
            lines.append(f"- {ev}")
        lines.append("")

    if diagnosis.recommended_actions:
        lines.append("**Recommended Actions:**")
        lines.append("")
        lines.append("| Priority | Action | Est. Downtime |")
        lines.append("|----------|--------|---------------|")
        for action in diagnosis.recommended_actions:
            pri = _priority_badge(action.get("priority", ""))
            act = action.get("action", "")
            dt = action.get("estimated_downtime_hours", "—")
            lines.append(f"| {pri} | {act} | {dt}h |")
        lines.append("")

    if diagnosis.parts_at_risk:
        lines.append(f"**Parts at Risk:** {', '.join(diagnosis.parts_at_risk)}")
        lines.append("")

    if diagnosis.pattern_notes:
        lines.append(f"**Pattern Notes:** {diagnosis.pattern_notes}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def generate_fleet_section(summary: FleetSummary) -> str:
    """Generate the fleet summary section."""
    lines = []
    lines.append("## Fleet Summary")
    lines.append("")
    lines.append(f"> {summary.executive_summary}")
    lines.append("")

    if summary.top_issues:
        lines.append("### Top Issues")
        lines.append("")
        for i, issue in enumerate(summary.top_issues, 1):
            machines = ", ".join(issue.get("affected_machines", []))
            lines.append(f"**{i}. {issue.get('issue', '')}**")
            lines.append(f"- Affected: `{machines}`")
            lines.append(f"- Frequency: {issue.get('frequency', 'N/A')}")
            lines.append(f"- Impact: {issue.get('business_impact', 'N/A')}")
            lines.append("")

    if summary.maintenance_priorities:
        lines.append("### Maintenance Priorities")
        lines.append("")
        lines.append("| Priority | Action | Machines | Rationale |")
        lines.append("|----------|--------|----------|-----------|")
        for mp in summary.maintenance_priorities:
            pri = _priority_badge(mp.get("priority", ""))
            machines = ", ".join(mp.get("machines", []))
            lines.append(
                f"| {pri} | {mp.get('action', '')} | `{machines}` | {mp.get('rationale', '')} |"
            )
        lines.append("")

    if summary.trend_alerts:
        lines.append("### Trend Alerts")
        lines.append("")
        for alert in summary.trend_alerts:
            lines.append(f"- ⚠️ {alert}")
        lines.append("")

    if summary.estimated_downtime_savings_hours:
        lines.append(
            f"**Estimated Downtime Savings:** "
            f"{summary.estimated_downtime_savings_hours:.1f} hours "
            f"(if all recommended actions are completed)"
        )
        lines.append("")

    return "\n".join(lines)


def generate_report(
    diagnoses: list[DiagnosisResult],
    fleet_summary: FleetSummary | None = None,
    stats: dict | None = None,
) -> str:
    """
    Generate a complete diagnostic report in Markdown.

    Args:
        diagnoses: List of individual failure diagnoses.
        fleet_summary: Optional fleet-level summary.
        stats: Optional dict with dataset statistics.

    Returns:
        Complete Markdown report as a string.
    """
    lines = []
    lines.append("# Manufacturing Log Diagnostic Report")
    lines.append("")
    lines.append(
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"| Powered by Claude (Anthropic)*"
    )
    lines.append("")

    # Dataset stats
    if stats:
        lines.append("## Dataset Overview")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for key, val in stats.items():
            lines.append(f"| {key} | {val} |")
        lines.append("")

    # Fleet summary
    if fleet_summary:
        lines.append(generate_fleet_section(fleet_summary))

    # Individual diagnoses
    lines.append("## Detailed Failure Analysis")
    lines.append("")
    for diagnosis in diagnoses:
        lines.append(generate_diagnosis_section(diagnosis))

    return "\n".join(lines)


def save_report(report: str, path: str | Path) -> Path:
    """Save report to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"Report saved → {path}")
    return path
