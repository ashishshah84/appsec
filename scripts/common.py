"""Shared helpers for the AI AppSec agent scripts.

Kept deliberately small and transparent: students should be able to read this
file and understand exactly how the demo talks to Claude, how structured
results are saved, and how job outputs are passed between GitHub Actions jobs.

Nothing here is hidden inside a third-party action.
"""

from __future__ import annotations

import json
import os
import re
# `subprocess` is only used to run fixed `git` argv lists (never a shell); see run().
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any, NoReturn, Optional

# --- Paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PROMPTS_DIR = REPO_ROOT / "prompts"

# The model is configurable via the GEMINI_MODEL repo variable / env var so
# the demo is not pinned to a single version. This default is only a fallback.
# Gemini 2.0 Flash is used because it's covered by Google AI Studio's free,
# no-credit-card tier -- see https://ai.google.dev/pricing
DEFAULT_MODEL = "gemini-2.0-flash"

# Generous but bounded: the fixer returns full (small) file contents.
DEFAULT_MAX_TOKENS = 16000


# --- Logging ---------------------------------------------------------------

def log(message: str) -> None:
    """Print a message to the GitHub Actions log (stdout)."""
    print(message, flush=True)


def die(message: str, code: int = 1) -> NoReturn:
    """Print an error and exit non-zero (fail the job honestly)."""
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


# --- Prompt / artifact IO --------------------------------------------------

def read_prompt(filename: str) -> str:
    """Load an agent prompt from the prompts/ directory."""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def read_artifact(filename: str, default: str = "") -> str:
    """Read a text artifact, returning `default` if it does not exist."""
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def write_artifact(filename: str, content: str) -> Path:
    """Write a text artifact and return its path."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    log(f"Wrote artifact: {path.relative_to(REPO_ROOT)}")
    return path


def read_json_artifact(filename: str, default: Optional[dict] = None) -> Optional[dict]:
    """Read a JSON artifact, returning `default` if missing or unparsable."""
    raw = read_artifact(filename, "")
    if not raw.strip():
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def write_json_artifact(filename: str, data: Any) -> Path:
    """Write a JSON artifact (pretty-printed, sorted keys)."""
    return write_artifact(filename, json.dumps(data, indent=2, sort_keys=True) + "\n")


# --- GitHub Actions job outputs --------------------------------------------

def set_output(name: str, value: str) -> None:
    """Set a GitHub Actions step output via the $GITHUB_OUTPUT file.

    Job outputs are how we pass small structured decisions (e.g. finding=true)
    between jobs. Larger data travels as uploaded artifacts, not as outputs.
    """
    output_file = os.environ.get("GITHUB_OUTPUT")
    line = f"{name}={value}"
    if output_file:
        with open(output_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    log(f"::output:: {line}")


# --- Subprocess helper -----------------------------------------------------

def run(cmd: list[str], check: bool = True) -> str:
    """Run a command and return its stdout (used for git operations).

    `cmd` is always an argv list (never a shell string) and `shell` defaults to
    False, so there is no shell-injection surface here.
    """
    result = subprocess.run(  # nosec B603
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        die(f"Command failed ({' '.join(cmd)}):\n{result.stderr}")
    return result.stdout


# --- Gemini (free tier) -----------------------------------------------------
#
# Swapped from the Anthropic Messages API to Google's Gemini API so this demo
# can run entirely on Google AI Studio's free, no-credit-card tier. This is
# the ONLY function in the whole repo that talks to an LLM -- every calling
# script (reviewer/fixer/verifier/summary) is unchanged, since they only
# depend on this function's signature: (system_prompt, user_content) -> str.

def get_model() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def call_claude(system_prompt: str, user_content: str,
                max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Send one request to Gemini and return the text response.

    Kept the name `call_claude` so every other script in the repo (reviewer,
    fixer, verifier, summary) needs zero changes -- they only import this one
    function from common.py.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:  # pragma: no cover - environment issue
        die("The 'google-genai' package is not installed. Run: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        die("GEMINI_API_KEY is not set. Configure it as a repository secret.")

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=get_model(),
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
    except Exception as exc:  # pragma: no cover - network dependent
        die(f"Gemini API error: {exc}")

    text = (response.text or "").strip()
    if not text:
        die("Gemini returned an empty response.")
    return text


# --- JSON extraction -------------------------------------------------------

def extract_json(text: str) -> dict:
    """Extract the first balanced JSON object from a model response.

    The prompts ask for raw JSON, but we defensively strip code fences and scan
    for a balanced object so a stray fence or trailing note does not break the
    pipeline.
    """
    cleaned = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences if present.
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    if start == -1:
        die(f"No JSON object found in model response:\n{text[:500]}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = cleaned[start:index + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError as exc:
                    die(f"Model returned invalid JSON: {exc}\n{snippet[:500]}")
    die(f"Unbalanced JSON in model response:\n{text[:500]}")
