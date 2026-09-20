"""
Inference/model_wrappers.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import base64
import math
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional

from Inference.parser import classify, parse_fixed
from Inference.serving_utils import ProviderUnavailable, call_timeout_s, notice, require_packages, resolve_secret, retry_call


YES_TOKENS = {"yes", "Yes", "YES", " yes", " Yes", " YES"}
NO_TOKENS = {"no", "No", "NO", " no", " No", " NO"}


def pil_to_data_url(img, quality: int) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _messages(spec: Dict, image, prompt: str, system_message: str) -> List[Dict]:
    content = prompt
    if image is not None:
        content = [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": prompt}]
    messages = []
    if spec["reasoning"]:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": content})
    return messages


def _confidence_from_logprobs(response) -> Optional[float]:
    try:
        top = response.choices[0].logprobs.content[0].top_logprobs
    except (AttributeError, IndexError, TypeError):
        return None
    p_yes = sum(math.exp(lp.logprob) for lp in top if lp.token in YES_TOKENS)
    p_no = sum(math.exp(lp.logprob) for lp in top if lp.token in NO_TOKENS)
    if p_yes + p_no <= 0:
        return None
    return float(p_yes / (p_yes + p_no))


class ServedModel:

    def __init__(self, model_key: str, spec: Dict, cfg: Dict):
        self.model_key, self.spec, self.cfg = model_key, spec, cfg
        self.serving = cfg["serving"]
        self.system_message = cfg["prompts"]["reasoning_system_message"]
        self.modality = spec["modality"]
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import httpx
        from openai import OpenAI
        base_url = resolve_secret(self.cfg, "llm_api_base_url")
        api_key = resolve_secret(self.cfg, "llm_api_key")
        if not base_url:
            raise RuntimeError("no serving base URL: set llm_api_base_url in config.yaml or the environment variable named by llm_api_base_url_env")
        conc = max(1, int(self.serving["concurrency"]))
        limits = httpx.Limits(max_connections=conc + 8, max_keepalive_connections=conc + 8)
        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY",
                              http_client=httpx.Client(verify=bool(self.serving["verify_ssl"]),
                                                       timeout=float(self.serving["timeout_s"]), limits=limits),
                              max_retries=0)
        print(f"[ServedModel] {self.model_key} -> '{self.spec['served_name']}' via {base_url}", flush=True)

    def _one(self, image, prompt: str, max_tokens: int) -> Dict:
        self._ensure_client()
        image_url = None if image is None else pil_to_data_url(image, int(self.serving["jpeg_quality"]))
        kwargs = dict(model=self.spec["served_name"], messages=_messages(self.spec, image_url, prompt, self.system_message),
                      max_tokens=int(max_tokens), temperature=float(self.cfg["generation"]["temperature"]))
        if not self.spec["reasoning"]:
            kwargs.update(logprobs=True, top_logprobs=int(self.serving["logprobs_top_k"]))

        def _create(timeout):
            if timeout is not None:
                return self._client.chat.completions.create(timeout=timeout, **kwargs)
            return self._client.chat.completions.create(**kwargs)

        t0 = time.time()
        resp = retry_call(_create, self.serving, call_timeout_s(self.serving, max_tokens), self.model_key)
        choice = resp.choices[0]
        raw = choice.message.content or ""
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(resp, "usage", None)
        parsed = parse_fixed(raw, self.spec["reasoning"], self.cfg["parser"])
        return {"raw_answer": raw, "parsed_answer": parsed,
                "answer_class": classify(raw, parsed, self.spec["reasoning"], finish, self.cfg["parser"]),
                "confidence": None if self.spec["reasoning"] else _confidence_from_logprobs(resp),
                "finish_reason": finish, "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None), "elapsed_s": round(time.time() - t0, 3)}

    def answer_batch(self, items: List[Dict], max_tokens: int) -> List[Optional[Dict]]:
        conc = max(1, int(self.serving["concurrency"]))
        out: List[Optional[Dict]] = [None] * len(items)

        def _do(i):
            try:
                return i, self._one(items[i]["image"], items[i]["prompt"], max_tokens)
            except ProviderUnavailable as e:
                notice(f"abandon:{self.model_key}", f"[ServedModel] {self.model_key}: a call was abandoned ({e}); the case stays un-done.")
                return i, None

        with ThreadPoolExecutor(max_workers=conc) as ex:
            for i, res in ex.map(_do, range(len(items))):
                out[i] = res
        return out


class RadDinoModel:

    def __init__(self, model_key: str, spec: Dict, cfg: Dict, dataset: str):
        from models.raddino import RadDinoAnswerer
        self.model_key, self.spec, self.cfg = model_key, spec, cfg
        self.modality = "vision_only"
        self.answerer = RadDinoAnswerer(cfg, "chexpert" if dataset == "chexpert" else "mimic")
        self.fingerprint = self.answerer.fingerprint

    def answer_batch(self, items: List[Dict], max_tokens: int) -> List[Optional[Dict]]:
        t0 = time.time()
        probs = self.answerer.probabilities([it["image"] for it in items], [it["finding"] for it in items])
        out: List[Optional[Dict]] = []
        for p in probs:
            if p is None:
                out.append({"raw_answer": "probe_unavailable", "parsed_answer": -1, "answer_class": "unparsed", "confidence": None,
                            "finish_reason": None, "prompt_tokens": None, "completion_tokens": None, "elapsed_s": None})
                continue
            parsed = 1 if p >= 0.5 else 0
            out.append({"raw_answer": "Yes" if parsed else "No", "parsed_answer": parsed, "answer_class": "yes" if parsed else "no",
                        "confidence": p, "finish_reason": "stop", "prompt_tokens": None, "completion_tokens": None,
                        "elapsed_s": round((time.time() - t0) / max(1, len(items)), 4)})
        return out


def build_model(model_key: str, cfg: Dict, dataset: str):
    spec = cfg["models"][model_key]
    require_packages(["PIL"] + (["torch", "transformers"] if spec["backend"] == "raddino" else ["openai", "httpx"]), f"build_model {model_key}")
    if spec["backend"] == "raddino":
        return RadDinoModel(model_key, spec, cfg, dataset)
    if spec["backend"] != "server":
        raise ValueError(f"{model_key}: unknown backend {spec['backend']!r}; the panel runs on the serving system or in-process RAD-DINO")
    return ServedModel(model_key, spec, cfg)
