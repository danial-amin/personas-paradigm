"""OpenAI Chat Completions `response_format` JSON Schemas for persona experiments.

Uses strict, methodology-aligned fields per paradigm instead of a single `profile` blob.
See: https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

from typing import Any


def _obj(
    properties: dict[str, Any],
    *,
    descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for key, spec in properties.items():
        if isinstance(spec, dict) and "type" in spec:
            entry = dict(spec)
        else:
            entry = {"type": spec} if isinstance(spec, str) else spec
        if descriptions and key in descriptions:
            entry = {**entry, "description": descriptions[key]}
        props[key] = entry
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


def _persona_schema(paradigm_id: str) -> dict[str, Any]:
    """JSON Schema for one persona object (strict)."""
    d: dict[str, str] = {}
    if paradigm_id == "goal-driven":
        return _obj(
            {
                "persona_index": "integer",
                "display_name": "string",
                "identity": "string",
                "end_goals": "string",
                "life_goals": "string",
                "experience_goals": "string",
                "key_behaviors": "string",
                "scenario_of_use": "string",
            },
            descriptions={
                "persona_index": "1-based index in this output batch.",
                "display_name": "Short label or full name.",
                "identity": "Name, age, gender, occupation, one-sentence bio (goal-directed design).",
                "end_goals": "2–3 actionable product-related goals.",
                "life_goals": "1–2 broader motivations.",
                "experience_goals": "1–2 desired feelings during use.",
                "key_behaviors": "3–4 behaviors linked to goals.",
                "scenario_of_use": "3–5 sentence use narrative with the product.",
            },
        )
    if paradigm_id == "engaging":
        return _obj(
            {
                "persona_index": "integer",
                "display_name": "string",
                "name_and_photo_description": "string",
                "personal_background": "string",
                "day_in_the_life": "string",
                "relationships_and_social_context": "string",
                "hopes_and_fears": "string",
                "attitude_toward_product": "string",
            },
            descriptions={
                "name_and_photo_description": "Full name plus brief visual description for image generation.",
                "personal_background": "3–4 sentences, third person.",
                "day_in_the_life": "4–6 sentences typical day.",
                "relationships_and_social_context": "Key people and influence on product attitudes.",
                "hopes_and_fears": "Aspirations and anxieties; tie at least one of each to the product domain.",
                "attitude_toward_product": "Beliefs, past experience, emotions about the product.",
            },
        )
    if paradigm_id == "role-based":
        return _obj(
            {
                "persona_index": "integer",
                "display_name": "string",
                "role_title_and_organization": "string",
                "role_responsibilities": "string",
                "workflows_and_tasks": "string",
                "domain_expertise": "string",
                "organizational_constraints": "string",
                "collaboration_points": "string",
                "brief_demographics": "string",
            },
            descriptions={
                "role_title_and_organization": "Job title / role and org type.",
                "role_responsibilities": "3–5 key professional responsibilities.",
                "workflows_and_tasks": "2–3 workflows involving the product, steps and tools.",
                "domain_expertise": "novice/intermediate/expert with justification.",
                "organizational_constraints": "2–3 constraints (compliance, budget, hierarchy).",
                "collaboration_points": "Other roles and interaction nature.",
                "brief_demographics": "Minimal name, age, years of experience.",
            },
        )
    if paradigm_id == "extreme":
        return _obj(
            {
                "persona_index": "integer",
                "display_name": "string",
                "identity_context": "string",
                "defining_extreme_characteristics": "string",
                "unique_needs_and_requirements": "string",
                "barriers_to_use": "string",
                "workarounds_and_adaptations": "string",
                "value_to_design_process": "string",
                "scenario_of_extreme_use": "string",
            },
            descriptions={
                "identity_context": "Name, age, biographical note on why this user is extreme/atypical.",
                "defining_extreme_characteristics": "1–2 edge-of-distribution attributes and product relevance.",
                "unique_needs_and_requirements": "3–4 needs from those characteristics.",
                "barriers_to_use": "2–3 concrete barriers.",
                "workarounds_and_adaptations": "How they cope today.",
                "value_to_design_process": "What designing for them reveals.",
                "scenario_of_extreme_use": "3–5 sentence narrative under challenging conditions.",
            },
        )
    if paradigm_id == "proto":
        return _obj(
            {
                "persona_index": "integer",
                "display_name": "string",
                "profile": "string",
            },
            descriptions={
                "profile": "Full preliminary persona: unified prose, all attribute areas, paragraph spacing.",
            },
        )
    # Unknown paradigm: generic split still beats one opaque blob
    return _obj(
        {
            "persona_index": "integer",
            "display_name": "string",
            "profile": "string",
        },
        descriptions={
            "profile": "Complete persona following the methodology in the system prompt.",
        },
    )


def openai_structured_response_format(paradigm_id: str, num_personas: int) -> dict[str, Any]:
    """`response_format` value for `client.chat.completions.create` (Structured Outputs)."""
    item_schema = _persona_schema(paradigm_id)
    root: dict[str, Any] = {
        "type": "object",
        "properties": {
            "personas": {
                "type": "array",
                "description": f"Exactly {num_personas} distinct personas from the dataset.",
                "minItems": num_personas,
                "maxItems": num_personas,
                "items": item_schema,
            }
        },
        "required": ["personas"],
        "additionalProperties": False,
    }
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in paradigm_id) or "personas"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"personas_{safe_name}",
            "strict": True,
            "schema": root,
        },
    }
