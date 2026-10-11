"""Per-run setting overrides, so one process can run experiments fairly.

A/B experiments ("same requests, model A vs model B", "retry limit 2 vs 5",
"memory on vs off") need those settings to differ BETWEEN runs that execute at
the same time. The globals in app.config can't do that, so a run's overrides
live in a ContextVar: set at the start of run_pipeline (inside that run's own
asyncio task) and automatically visible in everything the run does, including
worker threads started with asyncio.to_thread, without leaking into any other
concurrent run.

Supported overrides (all optional):
    model            Ollama model name for every generate call of the run
    max_qa_attempts  QA attempts per task before escalating (1-10)
    use_memory       False = long-term memory is neither read nor written
                     (task memory AND failure patterns)

Everything else (communication mode, framework, interactivity) is already a
per-run parameter of run_pipeline.
"""
import re
from contextvars import ContextVar, Token
from typing import Any, Optional

from app.config import settings

ALLOWED = ("model", "max_qa_attempts", "use_memory")
_MODEL_RE = re.compile(r"^[\w.:/@+-]{1,100}$")
_overrides: ContextVar[dict] = ContextVar("aiopsforge_run_overrides", default={})


def validate(overrides: Optional[dict]) -> dict:
    """Returns a cleaned copy of `overrides` or raises ValueError. Unknown keys
    are rejected (a typo'd experiment factor must not silently do nothing)."""
    out: dict[str, Any] = {}
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k not in ALLOWED:
            raise ValueError(f"unknown run override {k!r}; supported: {', '.join(ALLOWED)}")
        if k == "model":
            if not isinstance(v, str) or not _MODEL_RE.match(v):
                raise ValueError("model must be an Ollama model name such as 'qwen2.5-coder:3b'")
            out[k] = v
        elif k == "max_qa_attempts":
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 10:
                raise ValueError("max_qa_attempts must be an integer from 1 to 10")
            out[k] = v
        else:  # use_memory
            if not isinstance(v, bool):
                raise ValueError("use_memory must be true or false")
            out[k] = v
    return out


def push(overrides: Optional[dict]) -> Token:
    """Activates `overrides` for the current context. Pair with pop()."""
    return _overrides.set(validate(overrides))


def pop(token: Token) -> None:
    try:
        _overrides.reset(token)
    except ValueError:  # reset from a different context: the run's own task ended anyway
        pass


def current() -> dict:
    return dict(_overrides.get())


def model() -> str:
    return _overrides.get().get("model") or settings.ollama_model


def max_qa_attempts() -> int:
    return _overrides.get().get("max_qa_attempts") or settings.max_qa_attempts


def memory_enabled() -> bool:
    return _overrides.get().get("use_memory", True)
