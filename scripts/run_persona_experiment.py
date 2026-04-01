#!/usr/bin/env python3
"""
Run the full persona experiment. Loop order: dataset → paradigm → output format → model
(so each paradigm/format is run across all models before the next paradigm).

Usage (from repository root):
  cp .env.example .env   # then add API keys
  python scripts/run_persona_experiment.py
  python scripts/run_persona_experiment.py --config experiment_config.test.yaml

Config path: `experiment_config.yaml` by default, or `EXPERIMENT_CONFIG` / `--config`.
API keys (depending on enabled models): ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data_loaders import load_csv_dataset, load_transcript_dataset
from structured_json_schemas import openai_structured_response_format


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def _slug(s: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9._-]+", "-", s.strip())
    return x.strip("-")[:160] or "model"


def _resolve_config_path(arg: str | None) -> Path:
    raw = (arg or os.environ.get("EXPERIMENT_CONFIG") or "experiment_config.yaml").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"Missing config file: {p}")
    return p


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_prompt(rel_path: str) -> str:
    p = REPO_ROOT / "prompts" / rel_path
    return p.read_text(encoding="utf-8")


def _inject_persona_count(text: str, n: int) -> str:
    """Replace 'appropriate number' wording without breaking 'create an N personas'."""
    text = text.replace(
        "From that data, create an appropriate number of personas",
        f"From that data, create exactly {n} personas",
    )
    text = text.replace(
        "Create an appropriate number of personas",
        f"Create exactly {n} personas",
    )
    text = text.replace(
        "create an appropriate number of personas",
        f"create exactly {n} personas",
    )
    return text


def _build_system_prompt(
    paradigm_file: str,
    product: str,
    product_domain: str,
    num_personas: int,
    output_format: str,
) -> str:
    base = _read_prompt(paradigm_file)
    text = base.replace("[PRODUCT]", product)
    text = text.replace("[PRODUCT DOMAIN]", product_domain)
    text = _inject_persona_count(text, num_personas)
    if output_format == "structured":
        suffix = _read_prompt("structured-output-instructions.txt")
        suffix = suffix.replace("[NUM_PERSONAS]", str(num_personas))
        text = text + "\n\n" + suffix
    return text


def _user_message(
    bundle: dict[str, Any],
    num_personas: int,
    output_format: str,
    paradigm_id: str | None = None,
) -> str:
    # Proto personas: rapid sketches without research data (no dataset in the user message).
    if paradigm_id == "proto":
        intro = (
            f"Generate exactly {num_personas} preliminary proto-personas for the product "
            "described in the system prompt. Do not rely on an external dataset—use plausible "
            "assumptions for this product category.\n\n"
        )
        if output_format == "structured":
            intro += "Follow the structured JSON schema given in your instructions.\n\n"
        elif output_format == "narrative":
            intro += (
                "When there are multiple personas, write each persona as its own block. "
                "After each complete persona except the last, output a single line containing "
                "exactly <<<PERSONA_BREAK>>> then a blank line, then start the next persona. "
                "Do not merge two personas into one continuous paragraph block.\n\n"
            )
        return intro + "Produce the personas now."

    intro = (
        f"Here is the dataset ({bundle['total_rows']} rows / records total). "
        f"Generate exactly {num_personas} personas.\n\n"
    )
    if output_format == "structured":
        intro += "Follow the structured JSON schema given in your instructions.\n\n"
    elif output_format == "narrative":
        intro += (
            "When there are multiple personas, write each persona as its own block. "
            "After each complete persona except the last, output a single line containing "
            "exactly <<<PERSONA_BREAK>>> then a blank line, then start the next persona. "
            "Do not merge two personas into one continuous paragraph block.\n\n"
        )
    return (
        f"{intro}"
        f"--- DATA SAMPLE ---\n{bundle['data_sample']}\n\n"
        f"--- SUMMARY STATISTICS ---\n{bundle['data_stats']}\n\n"
        "Using the data above, generate the personas now."
    )


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _openai_use_max_completion_tokens(model_id: str, override: Any) -> bool:
    """Newer OpenAI chat models require max_completion_tokens instead of max_tokens."""
    if override is True:
        return True
    if override is False:
        return False
    m = model_id.lower()
    if "gpt-5" in m:
        return True
    if re.search(r"\bo[134](-|$|\.|/)", m):
        return True
    return False


def _call_anthropic(
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    *,
    retry_cfg: dict[str, Any] | None = None,
) -> str:
    import anthropic

    try:
        from anthropic import APIStatusError
    except ImportError:
        APIStatusError = Exception

    rc = retry_cfg or {}
    max_attempts = max(1, int(rc.get("max_attempts", 6)))
    initial_delay = float(rc.get("initial_delay_seconds", 15))
    backoff = float(rc.get("backoff_multiplier", 2))

    client = anthropic.Anthropic()
    for attempt in range(max_attempts):
        try:
            msg = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            block = msg.content[0]
            if block.type != "text":
                raise RuntimeError(f"Unexpected Anthropic block type: {block.type}")
            return block.text
        except APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code == 529 and attempt < max_attempts - 1:
                delay = initial_delay * (backoff**attempt)
                print(f"  Anthropic overloaded (529), retrying in {delay:.0f}s…")
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            err = str(e).lower()
            if attempt < max_attempts - 1 and (
                "529" in str(e) or "overloaded" in err
            ):
                delay = initial_delay * (backoff**attempt)
                print(f"  Anthropic overloaded (retrying in {delay:.0f}s): {e!s}")
                time.sleep(delay)
                continue
            raise


def _call_openai(
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    *,
    use_max_completion_tokens: bool,
    structured_paradigm_id: str | None = None,
    num_personas: int = 5,
    use_api_json_schema: bool = True,
) -> str:
    from openai import OpenAI

    client = OpenAI()
    kwargs: dict[str, Any] = {
        "model": model_id,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if use_max_completion_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    if structured_paradigm_id and use_api_json_schema:
        kwargs["response_format"] = openai_structured_response_format(
            structured_paradigm_id, num_personas
        )

    try:
        r = client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""
    except Exception as e:
        if structured_paradigm_id and kwargs.get("response_format"):
            err = str(e).lower()
            if any(
                x in err
                for x in (
                    "response_format",
                    "json_schema",
                    "structured",
                    "unsupported",
                    "invalid_request",
                )
            ):
                print(
                    f"  Warning: OpenAI structured outputs unavailable ({e!s}); "
                    "retrying without API schema (prompt-only JSON)."
                )
                kwargs.pop("response_format", None)
                r = client.chat.completions.create(**kwargs)
                return r.choices[0].message.content or ""
        raise


def _google_response_text(resp: Any) -> str:
    """Extract text from google-genai GenerateContentResponse."""
    t = getattr(resp, "text", None)
    if t:
        return t
    candidates = getattr(resp, "candidates", None) or []
    parts_out: list[str] = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            pt = getattr(part, "text", None)
            if pt:
                parts_out.append(pt)
    if parts_out:
        return "\n".join(parts_out)
    feedback = getattr(resp, "prompt_feedback", None)
    raise RuntimeError(f"Empty Gemini response (prompt_feedback={feedback!r})")


def _call_google(
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model_id,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return _google_response_text(resp)


def _dispatch_llm(
    provider: str,
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    *,
    openai_max_completion_tokens: bool | None = None,
    anthropic_retry_cfg: dict[str, Any] | None = None,
    structured_paradigm_id: str | None = None,
    num_personas: int = 5,
    openai_structured_outputs: bool = True,
) -> str:
    if provider == "anthropic":
        return _call_anthropic(
            model_id,
            system,
            user,
            max_tokens,
            temperature,
            retry_cfg=anthropic_retry_cfg,
        )
    if provider == "openai":
        use_mct = _openai_use_max_completion_tokens(
            model_id, openai_max_completion_tokens
        )
        return _call_openai(
            model_id,
            system,
            user,
            max_tokens,
            temperature,
            use_max_completion_tokens=use_mct,
            structured_paradigm_id=structured_paradigm_id,
            num_personas=num_personas,
            use_api_json_schema=openai_structured_outputs,
        )
    if provider == "google":
        return _call_google(model_id, system, user, max_tokens, temperature)
    raise ValueError(f"Unknown provider: {provider}")


def _load_dataset_bundle(cfg: dict[str, Any], ds: dict[str, Any]) -> dict[str, Any]:
    kind = ds["kind"]
    if kind == "csv":
        return load_csv_dataset(
            REPO_ROOT / ds["path"],
            int(ds.get("sample_max_rows", 1000)),
        )
    if kind == "transcript_csv":
        paths = [REPO_ROOT / p for p in ds["paths"]]
        return load_transcript_dataset(
            paths,
            id_column=ds.get("id_column", "transcript_id"),
            text_column=ds["text_column"],
            max_chars_per_transcript=int(ds.get("max_chars_per_transcript", 12000)),
            max_rows_per_file=int(ds.get("max_rows_per_file", 50)),
        )
    raise ValueError(f"Unknown dataset kind: {kind}")


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Run persona generation experiment.")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="YAML config path (default: EXPERIMENT_CONFIG env or experiment_config.yaml)",
    )
    args = parser.parse_args()
    config_path = _resolve_config_path(args.config)
    cfg = _load_config(config_path)
    try:
        config_rel = str(config_path.relative_to(REPO_ROOT))
    except ValueError:
        config_rel = str(config_path)
    print(f"Using config: {config_rel}")
    product = cfg["product"]
    product_domain = cfg.get("product_domain", "")
    num_personas = int(cfg.get("num_personas", 5))
    out_root = REPO_ROOT / cfg.get("output_dir", "outputs/experiments")
    out_root.mkdir(parents=True, exist_ok=True)
    temperature = float(cfg.get("temperature", 0.7))
    anthropic_retry_cfg: dict[str, Any] = dict(cfg.get("anthropic_retry") or {})
    openai_structured_outputs = bool(cfg.get("openai_structured_outputs", True))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = cfg["datasets"]
    paradigms: dict[str, str] = cfg["paradigms"]
    formats: list[str] = cfg["output_formats"]
    models: list[dict[str, Any]] = cfg["models"]

    # Preload dataset bundles
    bundles: dict[str, dict[str, Any]] = {}
    for ds in datasets_cfg:
        ds_id = ds["id"]
        print(f"Loading dataset {ds_id!r}…")
        bundles[ds_id] = _load_dataset_bundle(cfg, ds)

    manifest_path = out_root / "manifest.jsonl"

    active_models: list[dict[str, Any]] = []
    for model_entry in models:
        provider = model_entry["provider"]
        model_id = model_entry["model_id"]
        key_env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(provider)
        if key_env and not os.environ.get(key_env):
            print(f"Skipping {provider}/{model_id}: {key_env} not set.")
            continue
        active_models.append(model_entry)

    attempted = len(active_models)

    # dataset → paradigm → format → model (all models for one paradigm/format before next)
    for ds_id, bundle in bundles.items():
        for paradigm_id, prompt_file in paradigms.items():
            for fmt in formats:
                system = _build_system_prompt(
                    prompt_file,
                    product,
                    product_domain,
                    num_personas,
                    fmt,
                )
                user = _user_message(
                    bundle, num_personas, fmt, paradigm_id=paradigm_id
                )
                for model_entry in active_models:
                    provider = model_entry["provider"]
                    model_id = model_entry["model_id"]
                    max_tokens = int(model_entry.get("max_tokens", 8192))
                    openai_mct = model_entry.get("use_max_completion_tokens")
                    model_slug = _slug(model_id)
                    rel_name = (
                        f"{ds_id}__{paradigm_id}__{fmt}__{provider}__{model_slug}"
                    )
                    ext = "json" if fmt == "structured" else "txt"
                    out_path = run_dir / f"{rel_name}.{ext}"
                    print(f"Running {rel_name}…")
                    try:
                        raw = _dispatch_llm(
                            provider,
                            model_id,
                            system,
                            user,
                            max_tokens,
                            temperature,
                            openai_max_completion_tokens=openai_mct,
                            anthropic_retry_cfg=anthropic_retry_cfg,
                            structured_paradigm_id=(
                                paradigm_id if fmt == "structured" else None
                            ),
                            num_personas=num_personas,
                            openai_structured_outputs=openai_structured_outputs,
                        )
                        if fmt == "structured":
                            cleaned = _strip_json_fences(raw)
                            try:
                                parsed = json.loads(cleaned)
                                out_path.write_text(
                                    json.dumps(parsed, indent=2, ensure_ascii=False)
                                    + "\n",
                                    encoding="utf-8",
                                )
                            except json.JSONDecodeError:
                                out_path.write_text(raw + "\n", encoding="utf-8")
                                print(
                                    f"  Warning: saved raw text; JSON parse failed for {out_path.name}"
                                )
                        else:
                            out_path.write_text(raw + "\n", encoding="utf-8")

                        record = {
                            "run_id": run_id,
                            "dataset_id": ds_id,
                            "paradigm_id": paradigm_id,
                            "output_format": fmt,
                            "provider": provider,
                            "model_id": model_id,
                            "path": str(out_path.relative_to(REPO_ROOT)),
                            "ok": True,
                        }
                    except Exception as e:
                        print(f"  ERROR {rel_name}: {e}")
                        record = {
                            "run_id": run_id,
                            "dataset_id": ds_id,
                            "paradigm_id": paradigm_id,
                            "output_format": fmt,
                            "provider": provider,
                            "model_id": model_id,
                            "path": None,
                            "ok": False,
                            "error": str(e),
                        }
                    with open(manifest_path, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save run metadata
    meta = {
        "run_id": run_id,
        "config_path": config_rel,
        "output_dir": str(run_dir.relative_to(REPO_ROOT)),
        "models_with_keys": attempted,
        "loop_order": "dataset → paradigm → output_format → model",
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    if not active_models:
        print(
            "No API keys found for configured models. Set ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY, and/or GOOGLE_API_KEY in .env, then re-run."
        )
    print(f"Done. Artifacts under {run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
