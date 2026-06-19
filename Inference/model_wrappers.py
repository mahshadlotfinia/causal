"""
Inference/model_wrappers.py
Created May 22, 2026

Model registry and wrappers for all models.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import base64
import math
import os
import pickle
import re
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")



MODEL_REGISTRY = {
    "LLaVA-Med-7B": {
        "hf_id": "/data/models/converted/llava-med-v1.5-7b-hf",
        "vision": True, "local": False, "reasoning": False,
    },
    "LLaVA-Med-7B-Local": {
        "hf_id": "/data/models/converted/llava-med-v1.5-7b-hf",
        "vision": True, "local": True, "local_vlm": True, "reasoning": False,
    },
    "MedGemma-1.5-4B": {
        "hf_id": "google/medgemma-1.5-4b-it",
        "vision": True, "local": False, "reasoning": False,
    },
    "Gemma-4-26B": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "vision": True, "local": False, "reasoning": False,
    },
    "Qwen3-VL-32B": {
        "hf_id": "Qwen/Qwen3-VL-32B-Instruct",
        "vision": True, "local": False, "reasoning": False,
    },
    "Mistral-Small-4-119B": {
        "hf_id": "mistralai/Mistral-Small-4-119B-2603-NVFP4",
        "vision": True, "local": False, "reasoning": False,
    },
    "GPT-5": {
        "hf_id": "gpt-5",
        "vision": True, "local": False, "reasoning": True, "api_type": "openai",
    },
    "MedGemma-27B-text": {
        "hf_id": "google/medgemma-27b-it",
        "vision": False, "local": False, "reasoning": False,
    },
    "DeepSeek-R1-7B": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "vision": False, "local": False, "reasoning": True,
    },
    "DeepSeek-R1-70B": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "vision": False, "local": False, "reasoning": True,
    },
    "RAD-DINO": {
        "hf_id": "microsoft/rad-dino",
        "vision": True, "local": True, "reasoning": False,
    },
}



def _parse_answer(text: str, reasoning: bool = False) -> int:
    """Returns 1 (Yes), 0 (No), or -1 (ambiguous).
    Cleans BPE token artifacts (Ġ=space, Ċ=newline) that some servers
    return for models like DeepSeek-R1, then strips <think> blocks.
    For reasoning models the answer is appended at the end, so the last
    non-empty line is checked first before falling back to the full scan.
    """
    text = text.replace("Ġ", " ").replace("Ċ", "\n")
    if reasoning:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    if not text:
        return -1

    # For reasoning models check the last non-empty line first
    if reasoning:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            last = lines[-1].lower().rstrip(".,!?:;")
            if last in ("yes", "yeah", "correct", "true", "present", "positive"):
                return 1
            if last in ("no", "not", "absent", "negative", "false", "incorrect"):
                return 0

    first = text.split()[0].lower().rstrip(".,!?:;")
    if first in ("yes", "yeah", "correct", "true", "present", "positive"):
        return 1
    if first in ("no", "not", "absent", "negative", "false", "incorrect"):
        return 0
    lower = text.lower()
    has_yes = "yes" in lower[:60]
    has_no  = "no"  in lower[:60]
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return -1


def _pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _to_openai_responses_input(messages: list) -> list:
    """
    Convert Chat Completions-style messages into Responses API input.
    This preserves roles and prompt text, and only renames multimodal
    content item types for OpenAI's Responses API.
    """
    converted = []

    for msg in messages:
        content = msg["content"]

        if isinstance(content, list):
            new_content = []
            for part in content:
                if part.get("type") == "image_url":
                    new_content.append({
                        "type": "input_image",
                        "image_url": part["image_url"]["url"],
                    })
                elif part.get("type") == "text":
                    new_content.append({
                        "type": "input_text",
                        "text": part["text"],
                    })
                else:
                    new_content.append(part)

            converted.append({
                "role": msg["role"],
                "content": new_content,
            })

        else:
            converted.append({
                "role": msg["role"],
                "content": content,
            })

    return converted



class APIModelWrapper:
    """
    Calls one of the 10 non-local models via the OpenAI-compatible API.

    Vision models receive the image encoded as a base64 JPEG data URL.
    Text-only models receive the prompt string only.
    Confidence is extracted from first-token logprobs when the server
    supports them (vLLM / LiteLLM both do); falls back to string-based
    confidence otherwise.
    """

    def __init__(self, model_name: str, params: dict):
        import httpx
        from openai import OpenAI, AzureOpenAI

        info = MODEL_REGISTRY[model_name]
        cfg  = params["CausalAudit"]

        self.model_name   = model_name
        self.hf_id        = info["hf_id"]
        self.vision       = info["vision"]
        self.reasoning    = info["reasoning"]
        self.api_type     = info.get("api_type", info.get("api", "local_server"))
        self.modality     = "multimodal" if self.vision else "text_only"
        self.max_tokens   = int(
            cfg["reasoning_max_new_tokens"] if self.reasoning
            else cfg["max_new_tokens"]
        )

        api_type = self.api_type
        self.is_azure = False
        self.azure_deployment = None

        if api_type == "openai":
            # PhysioNet/Stanford credentialed image data may NOT be sent to the
            # public OpenAI API. Route the GPT family through the Azure OpenAI
            # Service instead, with human review and abuse-monitoring logging
            # disabled at the resource level. Azure addresses the model by its
            # deployment name rather than the model id.
            self.is_azure = True
            self.client = AzureOpenAI(
                azure_endpoint = cfg["azure_openai_endpoint"],
                api_key        = cfg["azure_openai_api_key"],
                api_version    = cfg["azure_openai_api_version"],
            )
            self.azure_deployment = cfg.get("azure_openai_deployment", self.hf_id)
        else:
            self.client = OpenAI(
                base_url=cfg["api_base_url"],
                api_key=cfg["api_key"],
                http_client=httpx.Client(verify=False),
            )
        print(f"[APIModelWrapper] Ready: {model_name} ({self.hf_id}) via {api_type}")

    def predict(
        self,
        image: Optional[Image.Image],
        prompt: str,
        finding: str = None,
    ) -> Tuple[str, int, float]:

        if self.vision and image is not None:
            b64 = _pil_to_base64(image)
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        messages = []
        if self.reasoning:
            # Force reasoning models to conclude with a single word.
            # Without this, DeepSeek-R1 generates thousands of tokens of
            # internal reasoning and may never reach a Yes/No answer within
            # the token budget.
            messages.append({
                "role": "system",
                "content": (
                    "You are a clinical decision support tool. "
                    "After your reasoning, you MUST end your response with "
                    "a single word on its own line: either Yes or No. "
                    "No other text after that word."
                ),
            })
        messages.append({"role": "user", "content": content})

        # OpenAI API and GPT-5 models do not support logprobs with image inputs
        use_logprobs = (
            self.api_type != "openai"
            and not self.is_azure
            and not self.hf_id.startswith("gpt-5")
        )

        if self.api_type == "openai":
            request_kwargs = dict(
                model=(self.azure_deployment if self.is_azure else self.hf_id),
                input=_to_openai_responses_input(messages),
                max_output_tokens=self.max_tokens,
            )
        else:
            request_kwargs = dict(
                model=self.hf_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0,
            )
            if use_logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = 5

        try:
            if self.api_type == "openai":
                response = self.client.responses.create(**request_kwargs)
                raw_text = response.output_text or ""
            else:
                response = self.client.chat.completions.create(**request_kwargs)
                raw_text = response.choices[0].message.content or ""
        except Exception as e:
            return str(e), -1, 0.5

        parsed   = _parse_answer(raw_text, self.reasoning)
        confidence = self._extract_confidence(response, parsed)
        return raw_text, parsed, confidence

    def _extract_confidence(self, response, parsed_answer: int) -> float:
        """
        Derive P(Yes) from the first-token logprobs returned by the API.
        Falls back to a deterministic value if logprobs are unavailable.
        """
        try:
            first_token_lps = response.choices[0].logprobs.content[0].top_logprobs
        except (AttributeError, IndexError, TypeError):
            # Server did not return logprobs
            if parsed_answer == 1:
                return 1.0
            if parsed_answer == 0:
                return 0.0
            return 0.5

        yes_variants = {"yes", "Yes", "YES", " yes", " Yes", " YES"}
        no_variants  = {"no",  "No",  "NO",  " no",  " No",  " NO"}

        p_yes, p_no = 0.0, 0.0
        for lp in first_token_lps:
            prob = math.exp(lp.logprob)
            if lp.token in yes_variants:
                p_yes += prob
            elif lp.token in no_variants:
                p_no += prob

        return float(p_yes / (p_yes + p_no + 1e-8))



class LocalVLMWrapper:
    """
    Loads a multimodal VLM directly from HuggingFace (or a local path) and
    runs inference on device. Used for LLaVA-Med-7B-Local and any other VLM
    that cannot or should not route through the API server.

    Uses LlavaForConditionalGeneration for LLaVA-style models; falls back
    to AutoModelForCausalLM with trust_remote_code for others.
    """

    def __init__(self, model_name: str, params: dict, dataset_type: str = "mimic"):
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            LlavaForConditionalGeneration,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        info = MODEL_REGISTRY[model_name]
        cfg  = params["CausalAudit"]

        self.model_name = model_name
        self.hf_id      = info["hf_id"]
        self.vision     = info["vision"]
        self.reasoning  = info["reasoning"]
        self.modality   = "multimodal" if self.vision else "text_only"
        self.max_tokens = int(
            cfg["reasoning_max_new_tokens"] if self.reasoning
            else cfg["max_new_tokens"]
        )

        print(f"[LocalVLMWrapper] Loading {model_name} from {self.hf_id} ...")

        load_kwargs = dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # LLaVA-style models need LlavaForConditionalGeneration
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.hf_id, **load_kwargs
            )
            self.is_llava = True
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_id, **load_kwargs
            )
            self.is_llava = False

        self.processor = AutoProcessor.from_pretrained(
            self.hf_id, trust_remote_code=True
        )

        self.model.eval()

        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Grab tokenizer for confidence extraction
        self.tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )

        print(f"[LocalVLMWrapper] {model_name} loaded (llava={self.is_llava}).")

    def _prepare_inputs(self, image: Optional[Image.Image], prompt: str) -> dict:
        if self.is_llava:
            formatted = f"USER: <image>\n{prompt}\nASSISTANT:"
            return self.processor(
                text=formatted,
                images=image,
                return_tensors="pt",
            )
        # Generic chat-template path
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]}]
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self.processor(text=[text], images=[image], return_tensors="pt")
        except Exception:
            return self.processor(text=prompt, images=image, return_tensors="pt")

    def predict(
        self,
        image: Optional[Image.Image],
        prompt: str,
        finding: str = None,
    ) -> Tuple[str, int, float]:
        inputs = self._prepare_inputs(image, prompt)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids    = output.sequences[:, prompt_len:]
        raw_text   = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        parsed     = _parse_answer(raw_text, self.reasoning)
        confidence = _compute_local_confidence(output.scores, self.tokenizer, parsed)
        return raw_text, parsed, confidence


def _compute_local_confidence(
    scores: Optional[tuple],
    tokenizer,
    parsed_answer: int,
) -> float:
    """Derives P(Yes) from first-token logits for locally loaded models."""
    if scores is None:
        return 1.0 if parsed_answer == 1 else (0.0 if parsed_answer == 0 else 0.5)

    logits = scores[0]
    probs  = torch.softmax(logits.float(), dim=-1).squeeze(0)

    yes_ids = [
        tid
        for tok in ["Yes", "yes", "YES", " Yes", " yes"]
        for tid in tokenizer.encode(tok, add_special_tokens=False)
        if tid < len(probs)
    ]
    no_ids = [
        tid
        for tok in ["No", "no", "NO", " No", " no"]
        for tid in tokenizer.encode(tok, add_special_tokens=False)
        if tid < len(probs)
    ]
    p_yes = probs[yes_ids].sum().item() if yes_ids else 0.0
    p_no  = probs[no_ids].sum().item()  if no_ids  else 0.0
    return float(p_yes / (p_yes + p_no + 1e-8))



class VisionOnlyWrapper:
    """
    RAD-DINO (loaded locally) + per-finding sklearn LogisticRegression probes.
    Probes must be trained via main_setup_raddino_probe() (MIMIC) or
    main_setup_raddino_probe_chexpert() before use.
    """

    def __init__(self, model_name: str, params: dict, dataset_type: str = "mimic"):
        from transformers import AutoModel, AutoProcessor

        cfg = params["CausalAudit"]

        self.model_name   = model_name
        self.hf_id        = MODEL_REGISTRY[model_name]["hf_id"]
        self.modality     = "vision_only"
        self.dataset_type = dataset_type

        probe_dir = (
            cfg["raddino_chexpert_probe_dir"]
            if dataset_type == "chexpert"
            else cfg["raddino_probe_dir"]
        )

        print(f"[VisionOnlyWrapper] Loading {model_name} from HuggingFace ...")

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = AutoModel.from_pretrained(
            self.hf_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        probe_file = os.path.join(probe_dir, "raddino_probes.pkl")
        if not os.path.exists(probe_file):
            raise FileNotFoundError(
                f"RAD-DINO probes not found at {probe_file}. "
                f"Run main_setup_raddino_probe() first."
            )
        with open(probe_file, "rb") as f:
            saved = pickle.load(f)
        self.probes  = saved["probes"]
        self.scaler  = saved["scaler"]
        print(f"[VisionOnlyWrapper] Loaded. Probes available: {list(self.probes.keys())}")

    def _extract_features(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        return self.scaler.transform(cls)

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        finding: str = None,
    ) -> Tuple[str, int, float]:
        if finding not in self.probes:
            return "probe_unavailable", -1, 0.5
        features = self._extract_features(image)
        proba    = self.probes[finding].predict_proba(features)[0]
        p_yes    = float(proba[1])
        parsed   = 1 if p_yes >= 0.5 else 0
        return ("Yes" if parsed == 1 else "No"), parsed, p_yes



def load_model_wrapper(model_name: str, params: dict, dataset_type: str = "mimic"):
    """Returns the appropriate wrapper instance for model_name."""
    assert model_name in MODEL_REGISTRY, (
        f"Unknown model '{model_name}'. "
        f"Available: {list(MODEL_REGISTRY.keys())}"
    )
    info = MODEL_REGISTRY[model_name]
    if info.get("local_vlm"):
        return LocalVLMWrapper(model_name, params, dataset_type=dataset_type)
    if info["local"]:
        return VisionOnlyWrapper(model_name, params, dataset_type=dataset_type)
    return APIModelWrapper(model_name, params)