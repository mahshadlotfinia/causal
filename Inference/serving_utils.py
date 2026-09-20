"""
Inference/serving_utils.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import random
import threading
import time
from typing import Callable, Dict, Optional, Tuple


_PLACEHOLDER_MARKERS = ("YOUR_", "_HERE")


def is_placeholder(value: Optional[str]) -> bool:
    return not value or any(m in str(value) for m in _PLACEHOLDER_MARKERS)


def resolve_secret(cfg: Dict, key: str) -> Optional[str]:
    value = cfg.get(key)
    if not is_placeholder(value):
        return value
    return os.environ.get(str(cfg.get(f"{key}_env", "")), None) or None


def require_packages(names, stage: str) -> None:
    import importlib
    missing = []
    for n in names:
        try:
            importlib.import_module(n)
        except ImportError:
            missing.append(n)
    if missing:
        raise RuntimeError(f"[{stage}] the environment lacks {missing}; run this line under the sorooshllm environment (see CLAUDE.md Section 1)")


class ProviderUnavailable(RuntimeError):

    deterministic = False


def is_rate_limit(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, openai.RateLimitError):
            return True
        if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", 0) == 429:
            return True
    except Exception:
        pass
    m = str(e).lower()
    return " 429" in m or "rate limit" in m


def retry_after_s(e: Exception) -> Optional[float]:
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                return float(ra)
    except Exception:
        pass
    return None


def is_timeout(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, openai.APITimeoutError):
            return True
    except Exception:
        pass
    try:
        import httpx
        if isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            return True
    except Exception:
        pass
    return isinstance(e, TimeoutError) or "timed out" in str(e).lower()


def is_transient_server_error(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)):
            return True
        if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", 0) in (502, 503, 504, 429):
            return True
    except Exception:
        pass
    try:
        import httpx
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
                          httpx.PoolTimeout, httpx.RemoteProtocolError)):
            return True
    except Exception:
        pass
    if isinstance(e, (ConnectionError, ConnectionRefusedError, TimeoutError)):
        return True
    m = str(e).lower()
    return any(s in m for s in ("connection refused", "connection error", "connection reset", "connect",
                                "timed out", "timeout", "temporarily unavailable", "reset by peer",
                                "bad gateway", "service unavailable", "gateway timeout",
                                " 502", " 503", " 504", " 429"))


_NOTICE_STATE: Dict[str, Tuple[float, int]] = {}
_NOTICE_LOCK = threading.Lock()


def notice(key: str, message: str, min_interval_s: float = 60.0) -> None:
    now = time.time()
    with _NOTICE_LOCK:
        last, suppressed = _NOTICE_STATE.get(key, (0.0, 0))
        if last and (now - last) < min_interval_s:
            _NOTICE_STATE[key] = (last, suppressed + 1)
            return
        _NOTICE_STATE[key] = (now, 0)
    extra = f" [{suppressed} further occurrence(s) suppressed in the last {now - last:.0f}s]" if suppressed else ""
    print(message + extra, flush=True)


def call_timeout_s(serving: Dict, max_tokens: int) -> float:
    return min(float(serving["timeout_request_max_s"]),
               float(serving["timeout_s"]) + int(max_tokens or 0) * float(serving["timeout_per_token_s"]))


def retry_call(fn: Callable[[Optional[float]], object], serving: Dict, call_timeout: float, name: str):
    max_retries = max(1, int(serving["max_retries"]))
    base = float(serving["retry_backoff_s"])
    cap = float(serving["retry_backoff_max_s"])
    connect_budget = float(serving["connect_retry_max_s"])
    t0 = time.time()
    last, attempt, rl_attempt, announced = None, 0, 0, False
    slept, long_tries, timed_out = 0.0, 0, False
    while True:
        t_attempt = time.time()
        try:
            try:
                return fn(call_timeout)
            except TypeError:
                return fn(None)
        except Exception as e:
            last = e
            attempt_s = time.time() - t_attempt
            if is_rate_limit(e):
                rl_attempt += 1
                if rl_attempt > int(serving["rate_limit_retries"]):
                    break
                wait = retry_after_s(e) or min(30.0, base * (2 ** min(rl_attempt, 5)))
                if not announced:
                    notice(f"429:{name}", f"[{name}] throttled (429). Backing off; lower serving.concurrency if this persists.")
                    announced = True
                time.sleep(wait * (0.7 + 0.6 * random.random()))
                continue
            if is_transient_server_error(e):
                if not announced:
                    notice(f"outage:{name}", f"[{name}] call failed after {attempt_s:.0f}s ({type(e).__name__}); waiting for the server.")
                    announced = True
                if is_timeout(e) and call_timeout and attempt_s >= 0.9 * float(call_timeout):
                    timed_out = True
                    break
                if call_timeout and attempt_s >= 0.5 * float(call_timeout):
                    long_tries += 1
                    if long_tries >= max_retries:
                        break
                if slept >= connect_budget:
                    break
                sleep = min(cap, base * (2 ** min(attempt, 8))) * (0.7 + 0.6 * random.random())
                time.sleep(sleep)
                slept += sleep
                attempt += 1
                continue
            raise
    exc = ProviderUnavailable(f"[{name}] abandoned after {time.time() - t0:.0f}s "
                              f"({'spent its full per-call budget' if timed_out else f'{slept:.0f}s of backoff'}): {last}")
    exc.deterministic = timed_out
    raise exc from last
