"""
LLM-powered failure analysis — multi-provider support.

Supports:
  - Google Gemini  (free — recommended to start)
  - Groq           (free tier — fast inference)
  - Ollama         (free — fully local, no API key)
  - Anthropic      (paid — Claude)
  - OpenAI         (paid — GPT)

Takes structured failure windows from the parser and uses an LLM to:
1. Classify the root cause and failure mode
2. Assess severity and urgency
3. Generate actionable maintenance recommendations
4. Identify cross-machine patterns
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Prompt templates (shared across all providers)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert manufacturing engineer specializing in laser cutting \
and welding equipment diagnostics. You analyze machine logs, sensor telemetry, and \
operator notes to identify failure modes, root causes, and recommend corrective actions.

You have deep knowledge of:
- Laser resonator and optical path systems
- CNC motion control and mechanical drive trains
- Assist gas delivery and pressure regulation
- Thermal management and coolant systems
- Material science and cutting parameter optimization

Always respond with structured JSON matching the requested schema. \
Be specific and actionable in your recommendations. Reference actual sensor \
values and operator observations in your reasoning."""

SINGLE_WINDOW_PROMPT = """Analyze the following failure window from machine {machine_id}.

**Failure Window:**
- Time range: {start} → {end} ({duration_minutes} min)
- Events: {event_count}
- Error codes: {error_codes}

**Log entries (chronological):**
{entries_json}

Respond with ONLY valid JSON in this exact schema:
{{
  "root_cause": "Brief description of the most likely root cause",
  "failure_mode": "One of: THERMAL | OPTICAL | PNEUMATIC | MECHANICAL | MATERIAL | ELECTRICAL | UNKNOWN",
  "confidence": 0.0 to 1.0,
  "severity_assessment": "One of: LOW | MEDIUM | HIGH | CRITICAL",
  "evidence": [
    "Specific observation from the logs supporting this diagnosis"
  ],
  "recommended_actions": [
    {{
      "action": "What to do",
      "priority": "IMMEDIATE | NEXT_SHIFT | SCHEDULED",
      "estimated_downtime_hours": 0.0
    }}
  ],
  "parts_at_risk": ["Component names that may need replacement"],
  "pattern_notes": "Any notable patterns in the sensor data or timing"
}}"""

BATCH_SUMMARY_PROMPT = """You have analyzed {num_windows} failure windows across the fleet. \
Here are the individual diagnoses:

{diagnoses_json}

Provide a fleet-level summary. Respond with ONLY valid JSON:
{{
  "executive_summary": "2-3 sentence overview of fleet health",
  "top_issues": [
    {{
      "issue": "Description",
      "affected_machines": ["machine IDs"],
      "frequency": "How often this occurs",
      "business_impact": "Production impact"
    }}
  ],
  "maintenance_priorities": [
    {{
      "action": "What to do",
      "machines": ["machine IDs"],
      "priority": "IMMEDIATE | THIS_WEEK | THIS_MONTH",
      "rationale": "Why this matters"
    }}
  ],
  "trend_alerts": [
    "Any emerging patterns that could become problems"
  ],
  "estimated_downtime_savings_hours": 0.0
}}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    """Result of analyzing a single failure window."""
    machine_id: str
    start: str
    end: str
    root_cause: str
    failure_mode: str
    confidence: float
    severity_assessment: str
    evidence: list[str]
    recommended_actions: list[dict]
    parts_at_risk: list[str]
    pattern_notes: str
    raw_response: dict


@dataclass
class FleetSummary:
    """Result of fleet-level analysis."""
    executive_summary: str
    top_issues: list[dict]
    maintenance_priorities: list[dict]
    trend_alerts: list[str]
    estimated_downtime_savings_hours: float
    raw_response: dict


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Send a prompt and return the raw text response."""
        ...


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(b.text for b in message.content if b.type == "text")


class OpenAICompatibleBackend(LLMBackend):
    """
    OpenAI-compatible backend.

    Also works with any OpenAI-compatible API:
      - Groq:     base_url="https://api.groq.com/openai/v1"
      - Ollama:   base_url="http://localhost:11434/v1"
      - Together:  base_url="https://api.together.xyz/v1"
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None):
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


class GeminiBackend(LLMBackend):
    """Google Gemini API backend (free tier: 15 RPM, 1M tokens/day)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1:8b",
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o-mini",
}

ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": None,
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

INSTALL_HINTS = {
    "gemini": "pip install google-genai",
    "groq": "pip install openai",
    "ollama": "pip install openai  # + install Ollama from https://ollama.com",
    "anthropic": "pip install anthropic",
    "openai": "pip install openai",
}

FREE_KEY_URLS = {
    "gemini": "https://aistudio.google.com/apikey",
    "groq": "https://console.groq.com/keys",
}


def create_backend(
    provider: str = "gemini",
    api_key: str | None = None,
    model: str | None = None,
) -> LLMBackend:
    """
    Create an LLM backend for the given provider.

    Args:
        provider: One of "gemini", "groq", "ollama", "anthropic", "openai"
        api_key: API key (auto-detected from env vars if not provided)
        model: Model name (uses sensible default per provider if not provided)

    Returns:
        An LLMBackend instance ready to use.
    """
    provider = provider.lower()
    model = model or DEFAULT_MODELS.get(provider, "")

    # Resolve API key from env if not provided
    env_key = ENV_KEYS.get(provider)
    if env_key and not api_key:
        api_key = os.environ.get(env_key, "")

    if env_key and not api_key:
        free_url = FREE_KEY_URLS.get(provider, "")
        url_hint = f"\n  Get a free key: {free_url}" if free_url else ""
        raise ValueError(
            f"API key required for {provider}.\n"
            f"  Set {env_key} env var or pass --api-key.{url_hint}\n"
            f"  Install: {INSTALL_HINTS.get(provider, '')}"
        )

    if provider == "gemini":
        return GeminiBackend(api_key=api_key, model=model)

    elif provider == "groq":
        return OpenAICompatibleBackend(
            api_key=api_key,
            model=model,
            base_url="https://api.groq.com/openai/v1",
        )

    elif provider == "ollama":
        return OpenAICompatibleBackend(
            api_key="ollama",  # Ollama doesn't check the key
            model=model,
            base_url="http://localhost:11434/v1",
        )

    elif provider == "anthropic":
        return AnthropicBackend(api_key=api_key, model=model)

    elif provider == "openai":
        return OpenAICompatibleBackend(api_key=api_key, model=model)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: gemini, groq, ollama, anthropic, openai"
        )


# ---------------------------------------------------------------------------
# Analyzer (provider-agnostic)
# ---------------------------------------------------------------------------

class ManufacturingAnalyzer:
    """LLM-powered manufacturing log analyzer."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the analyzer.

        Args:
            provider: LLM provider — "gemini" (free), "groq" (free),
                      "ollama" (free/local), "anthropic", or "openai"
            api_key: API key (auto-detected from env if not set)
            model: Model override (uses best default per provider)
        """
        self.provider = provider
        self.backend = create_backend(provider, api_key, model)

    def _call_llm(self, user_prompt: str, max_tokens: int = 2000) -> dict:
        """Send a prompt to the LLM and parse the JSON response."""
        text = self.backend.call(SYSTEM_PROMPT, user_prompt, max_tokens)

        # Parse JSON — handle markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse LLM response: {e}", "raw_text": text}

    def analyze_window(self, window: dict) -> DiagnosisResult:
        """Analyze a single failure window."""
        entries_json = json.dumps(window["entries"], indent=2, default=str)

        prompt = SINGLE_WINDOW_PROMPT.format(
            machine_id=window["machine_id"],
            start=window["start"],
            end=window["end"],
            duration_minutes=window["duration_minutes"],
            event_count=window["event_count"],
            error_codes=", ".join(window["error_codes"]),
            entries_json=entries_json,
        )

        result = self._call_llm(prompt)

        if "error" in result:
            return DiagnosisResult(
                machine_id=window["machine_id"],
                start=window["start"],
                end=window["end"],
                root_cause=f"Analysis failed: {result['error']}",
                failure_mode="UNKNOWN",
                confidence=0.0,
                severity_assessment="UNKNOWN",
                evidence=[],
                recommended_actions=[],
                parts_at_risk=[],
                pattern_notes="",
                raw_response=result,
            )

        return DiagnosisResult(
            machine_id=window["machine_id"],
            start=window["start"],
            end=window["end"],
            root_cause=result.get("root_cause", "Unknown"),
            failure_mode=result.get("failure_mode", "UNKNOWN"),
            confidence=result.get("confidence", 0.0),
            severity_assessment=result.get("severity_assessment", "UNKNOWN"),
            evidence=result.get("evidence", []),
            recommended_actions=result.get("recommended_actions", []),
            parts_at_risk=result.get("parts_at_risk", []),
            pattern_notes=result.get("pattern_notes", ""),
            raw_response=result,
        )

    def analyze_batch(
        self,
        windows: list[dict],
        progress_callback=None,
    ) -> list[DiagnosisResult]:
        """Analyze multiple failure windows."""
        results = []
        for i, window in enumerate(windows):
            if progress_callback:
                progress_callback(i, len(windows), window["machine_id"])
            result = self.analyze_window(window)
            results.append(result)
        return results

    def generate_fleet_summary(
        self,
        diagnoses: list[DiagnosisResult],
    ) -> FleetSummary:
        """Generate a fleet-level summary from individual diagnoses."""
        diagnoses_data = []
        for d in diagnoses:
            diagnoses_data.append({
                "machine_id": d.machine_id,
                "time_range": f"{d.start} → {d.end}",
                "root_cause": d.root_cause,
                "failure_mode": d.failure_mode,
                "severity": d.severity_assessment,
                "recommended_actions": d.recommended_actions,
            })

        prompt = BATCH_SUMMARY_PROMPT.format(
            num_windows=len(diagnoses),
            diagnoses_json=json.dumps(diagnoses_data, indent=2),
        )

        result = self._call_llm(prompt, max_tokens=3000)

        if "error" in result:
            return FleetSummary(
                executive_summary=f"Summary generation failed: {result['error']}",
                top_issues=[],
                maintenance_priorities=[],
                trend_alerts=[],
                estimated_downtime_savings_hours=0,
                raw_response=result,
            )

        return FleetSummary(
            executive_summary=result.get("executive_summary", ""),
            top_issues=result.get("top_issues", []),
            maintenance_priorities=result.get("maintenance_priorities", []),
            trend_alerts=result.get("trend_alerts", []),
            estimated_downtime_savings_hours=result.get(
                "estimated_downtime_savings_hours", 0
            ),
            raw_response=result,
        )
