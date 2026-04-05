import asyncio
import base64
import io
import json
import os
from typing import Any, Dict, List
from urllib import error, request

import numpy as np
from PIL import Image

try:
    import requests
except Exception:
    requests = None


XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MODELS = [
    "grok-4-1-fast-reasoning",
    "grok-4",
    "grok-4-1-fast",
    "grok-4-fast",
    "custom",
]
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_DETAIL = "high"


class GrokVisionLLM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Answer the prompt precisely.",
                    },
                ),
                "model": (
                    DEFAULT_MODELS,
                    {
                        "default": "grok-4-1-fast-reasoning",
                    },
                ),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a precise assistant. Return only the requested result text.",
                    },
                ),
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 180,
                        "min": 5,
                        "max": 3600,
                        "step": 1,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze"
    CATEGORY = "Grok"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    async def analyze(
        self,
        prompt: str,
        model: str,
        max_output_tokens: int,
        system_prompt: str,
        custom_model: str,
        timeout_seconds: int,
        api_key: str,
        image=None,
    ):
        api_key_value = self._clean_api_key(str(api_key or "").strip() or os.getenv("XAI_API_KEY", "").strip())
        if not api_key_value:
            raise ValueError(
                "No xAI API key found. Enter api_key in the node or set XAI_API_KEY in the environment."
            )

        resolved_model = self._resolve_model(model, custom_model)
        prompt = prompt or "Answer the prompt precisely."
        system_prompt = system_prompt or ""

        message_content: List[Dict[str, Any]] = []

        if image is not None:
            for image_index in range(self._batch_size(image)):
                image_data_uri = self._tensor_to_data_uri(image[image_index])
                message_content.append(
                    {
                        "type": "input_image",
                        "image_url": image_data_uri,
                        "detail": DEFAULT_DETAIL,
                    }
                )

        message_content.append(
            {
                "type": "input_text",
                "text": prompt,
            }
        )

        payload_input: List[Dict[str, Any]] = []
        if system_prompt.strip():
            payload_input.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        payload_input.append(
            {
                "role": "user",
                "content": message_content,
            }
        )

        payload: Dict[str, Any] = {
            "model": resolved_model,
            "input": payload_input,
            "max_output_tokens": int(max_output_tokens),
            "store": False,
        }

        response_json = await asyncio.to_thread(
            self._post_json,
            url=XAI_RESPONSES_URL,
            payload=payload,
            api_key=api_key_value,
            timeout_seconds=int(timeout_seconds),
        )

        text = self._extract_output_text(response_json)
        if not text:
            text = json.dumps(response_json, indent=2, ensure_ascii=False)

        print(f"[Grok Vision LLM] {text}")
        return (text,)

    @staticmethod
    def _clean_api_key(api_key: str) -> str:
        api_key = str(api_key or "").strip().strip('"').strip("'")
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:].strip()
        return api_key

    @staticmethod
    def _resolve_model(model: str, custom_model: str) -> str:
        selected = str(model or "").strip()
        custom_value = str(custom_model or "").strip()
        if selected == "custom":
            if not custom_value:
                raise ValueError("model is set to custom, but custom_model is empty.")
            return custom_value
        return selected or custom_value or "grok-4-1-fast-reasoning"

    @staticmethod
    def _batch_size(image) -> int:
        if image is None:
            return 0
        if getattr(image, "ndim", 0) == 3:
            return 1
        return int(image.shape[0])

    @staticmethod
    def _tensor_to_data_uri(image_tensor) -> str:
        if hasattr(image_tensor, "detach"):
            np_image = image_tensor.detach().cpu().numpy()
        else:
            np_image = np.asarray(image_tensor)

        np_image = np.clip(np_image, 0.0, 1.0)
        np_image = (np_image * 255.0).round().astype(np.uint8)

        if np_image.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims [H, W, C], got shape {np_image.shape}.")
        if np_image.shape[-1] not in (3, 4):
            raise ValueError(
                f"Expected 3 or 4 channels in last dimension, got shape {np_image.shape}."
            )

        pil_image = Image.fromarray(np_image)

        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format="PNG")
        raw_bytes = image_bytes.getvalue()

        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Encoded image is {len(raw_bytes)} bytes, which exceeds xAI's 20 MiB limit."
            )

        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        return f"data:image/{DEFAULT_IMAGE_FORMAT};base64,{b64}"

    @staticmethod
    def _extract_output_text(response_json: Dict[str, Any]) -> str:
        texts: List[str] = []

        output = response_json.get("output") or []
        for item in output:
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    texts.append(content["text"])

        if texts:
            return "\n\n".join(texts).strip()

        if isinstance(response_json.get("output_text"), str):
            return response_json["output_text"].strip()

        return ""

    @staticmethod
    def _headers(api_key: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-GrokVisionLLM/1.5",
        }

    def _post_json(self, url: str, payload: Dict[str, Any], api_key: str, timeout_seconds: int) -> Dict[str, Any]:
        headers = self._headers(api_key)

        if requests is not None:
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
                if response.status_code >= 400:
                    self._raise_http_error(response.status_code, response.text)
                return response.json()
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                if hasattr(exc, "response") and exc.response is not None:
                    self._raise_http_error(exc.response.status_code, exc.response.text)

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except error.HTTPError as http_error:
            try:
                body = http_error.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(http_error)
            self._raise_http_error(http_error.code, body)
        except error.URLError as url_error:
            raise RuntimeError(f"xAI API request failed: {url_error}") from url_error

    @staticmethod
    def _raise_http_error(status_code: int, body: str) -> None:
        message = body.strip() or "No response body"
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                message = (
                    parsed.get("error")
                    or parsed.get("message")
                    or parsed.get("code")
                    or json.dumps(parsed, ensure_ascii=False)
                )
        except Exception:
            pass

        if status_code == 403 and "1010" in message:
            raise RuntimeError(
                "xAI API returned HTTP 403 / 1010. This usually means the request was blocked upstream. "
                "Verify the API key is raw xai-... without 'Bearer ', and test the same host with curl."
            )

        raise RuntimeError(f"xAI API returned HTTP {status_code}: {message}")
