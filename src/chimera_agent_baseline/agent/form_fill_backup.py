"""Terminal form-fill node — prompt + parse against the output schema.

The ReAct loop ends when the model stops issuing tool calls. The router
then routes to this node, which prompts the same model to emit a JSON
object matching a per-case *judgment* shape (built dynamically from
:mod:`chimera_agent_baseline.output.schema`) and validates with
:class:`langchain_core.output_parsers.PydanticOutputParser`. The judgment
is then merged with the programmatic case fields (``case_id``, ``patient``,
``reveal_sequence`` — derived from the run itself) into the full
``Task<N>Output`` record.

We deliberately avoid ``model.with_structured_output`` — it relies on
function-calling support that varies wildly across providers (Gemma 4's
offline tool-call parser, for instance, sometimes does not emit a tool
call when there is only one forced schema-tool). Prompt-and-parse is
provider-neutral: any LangChain ``BaseChatModel`` works.

The node retries on validation errors up to ``max_retries`` times and
raises if every attempt fails — there is no partial / stub fallback, so
an unfillable case aborts the run loudly rather than writing a
half-formed prediction.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser

from chimera_agent_baseline.output.schema import (
    VARIABLES_BY_TASK,
    assemble_full_output,
    build_dynamic_model,
    eligible_variables,
    reveal_info_for_tool,
)

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are filling out the structured decision form for the prostate-cancer "
    "case you just analysed. The user message contains your reasoning "
    "transcript and the tools you called. Output a SINGLE JSON object matching "
    "the supplied schema EXACTLY — no extra keys, no markdown fences, no "
    "commentary before or after the JSON."
)


def make_form_fill_node(model: BaseChatModel, max_retries: int = 3):
    """Return a LangGraph node closure that captures the unbound *model*.

    *max_retries* is the number of validation attempts before the node
    raises :class:`RuntimeError`. A successful attempt yields a payload
    that already validates against the per-task output model
    (Task1Output / Task2Output / Task3Output).
    """

    def form_fill(state: dict[str, Any]) -> dict[str, Any]:
        messages = state["messages"]
        task = int(state.get("task", 1))
        case_id = state.get("case_id", "unknown")

        tool_order = _called_tools_in_order(messages)
        called = set(tool_order)
        transcript = _final_assistant_text(messages)
        elig = eligible_variables(task, called) if task in VARIABLES_BY_TASK else []

        Dynamic = build_dynamic_model(task, called)
        parser = PydanticOutputParser(pydantic_object=Dynamic)
        skeleton = _build_skeleton_instructions(task, elig)

        log.debug(
            "form_fill: case=%s task=%d called_tools=%s eligible_vars=%s",
            case_id,
            task,
            tool_order,
            elig,
        )

        base_user = _user_prompt(case_id, task, transcript, tool_order, elig) + "\n\n" + skeleton

        # Provider-agnostic retry loop. We keep the conversation flat — a
        # single system + user pair, regenerated on retry — because small
        # models can be derailed by long error-laden histories. The retry
        # message names the missing/invalid fields explicitly.
        warnings: list[str] = []
        judgment: dict[str, Any] | None = None
        retry_hint: str | None = None

        for attempt in range(1, max_retries + 1):
            user_content = base_user if not retry_hint else f"{base_user}\n\n{retry_hint}"
            try:
                response = model.invoke(
                    [
                        SystemMessage(content=_SYSTEM_PROMPT),
                        HumanMessage(content=user_content),
                    ]
                )
                raw = response.content if isinstance(response.content, str) else json.dumps(response.content)
                obj = parser.parse(_extract_json_object(raw))
                judgment = obj.model_dump(mode="json")
                break
            except Exception as exc:  # noqa: BLE001 — any parse/validation failure triggers a retry
                warnings.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                log.warning("form_fill parse failed (attempt %d) for %s: %s", attempt, case_id, exc)
                if attempt == max_retries:
                    break
                retry_hint = (
                    "Your previous attempt did not validate. The validation "
                    f"error was:\n{exc}\n\n"
                    "Emit the JSON object exactly matching the shape above. "
                    "Every required key must appear. Output ONLY the JSON."
                )

        if judgment is None:
            raise RuntimeError(
                f"form_fill: case {case_id}: all {max_retries} attempts failed to "
                f"produce schema-valid output. form_fill_warnings={warnings}"
            )

        reveal_sequence = _build_reveal_sequence(tool_order)
        patient = _build_patient(case_id, state)
        full = assemble_full_output(task, case_id, patient, reveal_sequence, judgment)
        return {"structured_response": full, "form_fill_warnings": warnings}

    return form_fill


# ---------------------------------------------------------------------------
# Programmatic case fields — not produced by the LLM.
# ---------------------------------------------------------------------------


def _called_tools_in_order(messages: list) -> list[str]:
    """Unique tool names in first-call order, derived from ``ToolMessage``s."""
    seen: set[str] = set()
    order: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name and m.name not in seen:
            seen.add(m.name)
            order.append(m.name)
    return order


def _build_reveal_sequence(tool_order: list[str]) -> list[dict[str, Any]]:
    """Build the ``reveal_sequence`` — one entry per tool call, in call order.

    Timestamps are synthetic (evenly spaced from "now"): the agent runtime
    does not currently record the wall-clock time of each tool call.
    """
    base_ts = datetime.now(timezone.utc)
    sequence: list[dict[str, Any]] = []
    for i, tool_name in enumerate(tool_order, start=1):
        key, label = reveal_info_for_tool(tool_name)
        ts = base_ts + timedelta(seconds=2 * (i - 1))
        sequence.append(
            {
                "page": "decision",
                "order": i,
                "key": key,
                "label": label,
                "value": "section",
                "via": "tool_call",
                "ts": ts.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            }
        )
    return sequence


def _build_patient(case_id: str, state: dict[str, Any]) -> dict[str, Any]:
    patient_state = state.get("patient") or {}
    return {
        "id": case_id,
        "psa_ng_ml": patient_state.get("psa"),
        "age_years": patient_state.get("age"),
    }


# ---------------------------------------------------------------------------
# Prompt builders + helpers
# ---------------------------------------------------------------------------


def _user_prompt(
    case_id: str,
    task: int,
    transcript: str,
    called_tools: list[str],
    eligible: list[str],
) -> str:
    tools_line = ", ".join(called_tools) if called_tools else "(no tools called)"
    head = (
        f"Case ID: {case_id}\n"
        f"Task: {task}\n\n"
        "Your reasoning transcript (final assistant message from the ReAct "
        'loop):\n"""\n'
        f"{transcript}\n"
        '"""\n\n'
        f"Tools you called during the ReAct loop: {tools_line}\n\n"
    )
    if task not in VARIABLES_BY_TASK:
        return head + (
            "Now fill out the form: predict whether biochemical recurrence will "
            "occur (event = 1) or not (event = 0 / censored by last follow-up), "
            "give your predicted months to recurrence (a non-negative number), "
            "and a focused reasoning naming the 2-4 factors that most influenced "
            "your estimate."
        )
    return head + (
        "Variables you may weight (you may ONLY weight these — every other "
        "variable was either out of scope for this task or behind a tool you "
        "did not call):\n"
        f"  {', '.join(eligible) if eligible else '(none)'}\n\n"
        "Now fill out the form. Weight each variable (not_used / noted / "
        "important / decisive); give an overall confidence; and a focused "
        "reasoning naming the 2-4 factors that most influenced your "
        "recommendation."
    )


def _build_skeleton_instructions(task: int, eligible: list[str]) -> str:
    """Concrete JSON-skeleton format instructions.

    PydanticOutputParser's ``get_format_instructions()`` dumps the full
    JSON schema (with ``$defs``, ``$ref``, ``additionalProperties``,
    etc.). Small models tested in the wild (Gemma 4 E2B) sometimes echo
    that schema back verbatim instead of producing an instance. A
    concrete shape with placeholder values is far more robust and is
    still unambiguous about which keys are required.
    """
    if task == 3:
        skeleton = "\n".join(
            [
                "{",
                '  "event": <0 | 1 — 1 if you predict biochemical recurrence will occur, '
                "0 if censored / no recurrence by last follow-up>,",
                '  "months_to_recurrence": <number — predicted months to recurrence / last follow-up>,',
                '  "repeat_test": <"<a short description of the recommended follow-up test>" | null>,',
                '  "free_text": "<at least 40 chars; the evidence and the 2-4 factors that drove your estimate>"',
                "}",
            ]
        )
        return (
            "Output a SINGLE JSON object matching exactly this shape (replace "
            "every <...> with a real value, or JSON null where indicated):\n\n"
            f"{skeleton}\n\n"
            "Do NOT emit any other keys. Do NOT wrap in markdown fences. Do NOT echo the schema."
        )

    weight_enum = '"not_used" | "noted" | "important" | "decisive"'
    confidence_enum = '"clear" | "borderline" | "uncertain"'

    weight_lines = [f'    "{var}": <{weight_enum}>,' for var in eligible]
    if weight_lines:
        weight_lines[-1] = weight_lines[-1].rstrip(",")

    if task == 1:
        decision_lines = ['  "biopsy_decision": <"yes" | "no">,']
    else:
        action_enum = '"active_surveillance" | "continued_surveillance" | "watchful_waiting" | "active_treatment"'
        decision_lines = [
            '  "treatment_recommendation": {',
            f'    "primary": <{action_enum}>,',
            '    "modalities": [<specific modality strings, or empty list>],',
            '    "detail": <"<free-text detail>" | null>,',
            '    "as_protocol": <"<surveillance protocol description>" | null — only when primary is a '
            "surveillance action, else null>,",
            '    "as_trigger": <"<trigger for escalation>" | null — only when primary is a surveillance '
            "action, else null>",
            "  },",
        ]

    skeleton = "\n".join(
        [
            "{",
            *decision_lines,
            f'  "confidence": <{confidence_enum}>,',
            '  "variable_weights": {',
            *weight_lines,
            "  },",
            '  "repeat_test": <"<a short description of the recommended follow-up>" | null>,',
            '  "free_text": "<at least 40 chars; name the 2-4 factors that drove your call>"',
            "}",
        ]
    )

    return (
        "Output a SINGLE JSON object matching exactly this shape (replace "
        "every <...> with a real value, or JSON null where indicated):\n\n"
        f"{skeleton}\n\n"
        f"`variable_weights` MUST include all and only these keys: {', '.join(eligible)}.\n"
        "Use the weight 'not_used' for any variable that did not influence your "
        "decision. Do NOT emit any other keys at the top level. Do NOT wrap in "
        "markdown fences. Do NOT echo the schema."
    )


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from raw model output."""
    if not text:
        return text
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _final_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            return m.content if isinstance(m.content, str) else json.dumps(m.content)
    return ""
