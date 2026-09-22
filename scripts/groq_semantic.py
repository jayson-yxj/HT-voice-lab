"""Groq strict-JSON semantic turn judge with local credential loading."""
import json
import getpass
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib import request
from urllib.error import HTTPError, URLError


SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["complete", "incomplete", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["status", "confidence", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """你负责判断多人现场对话中某一个说话人自己的当前发言是否在语义上讲完。
只依据这个人的文字内容判断。换人、静音、音频片段结束都不是讲完的证据。
完整表达了观点、问题或请求时选择 complete；句子未完、列举未完、明确还要继续时选择 incomplete；证据不足时选择 uncertain。
不要决定助手是否回复；回复权限由本地显式点名门控单独控制。"""


def _retry_after_seconds(headers, message, attempt):
    value = headers.get("Retry-After") if headers else None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        match = re.search(r"try again in ([0-9.]+)(ms|s)", message, re.IGNORECASE)
        if match:
            delay = float(match.group(1))
            return delay / 1000 if match.group(2).lower() == "ms" else delay
    return min(8.0, 0.5 * 2 ** attempt)


def key_path():
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "ht-voice-lab" / "groq.key"


def load_api_key():
    value = os.environ.get("GROQ_API_KEY")
    if value:
        return value
    path = key_path()
    if not path.is_file():
        return None
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("Groq key file permissions are too broad; run chmod 600 " + str(path))
    value = path.read_text().strip()
    return value or None


def login():
    value = getpass.getpass("Groq API Key: ").strip()
    if not value:
        raise ValueError("Groq API Key cannot be empty")
    path = key_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix="groq-key-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("Groq key saved locally with mode 600: " + str(path))
    return 0


def _error_message(exc):
    try:
        return json.loads(exc.read().decode(errors="replace"))["error"]["message"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return exc.reason


def _permission_hint(code):
    if code != 403:
        return ""
    return ("; check the selected Groq project and pending Model Terms at "
            "https://console.groq.com/settings/model-terms; if those are correct, "
            "contact Groq support because this account returned a generic permission denial")


def _open_json(opener, call, timeout):
    """Retry transient proxy/TLS failures without retrying HTTP responses."""
    for attempt in range(5):
        try:
            with opener.open(call, timeout=timeout) as response:
                return json.load(response)
        except HTTPError:
            raise
        except (URLError, TimeoutError) as exc:
            if attempt == 4:
                raise OSError("Groq network request failed after 5 attempts: {}".format(
                    exc)) from None
            time.sleep(_retry_after_seconds(None, "", attempt))


def available_models(timeout=10.0, api_key=None, proxy=None):
    """Validate Groq access before starting a long audio run."""
    key = api_key or load_api_key()
    if not key:
        raise ValueError("GROQ_API_KEY is not set; run './lab login-groq'")
    selected_proxy = proxy or os.environ.get("GROQ_HTTPS_PROXY")
    opener = (request.build_opener(request.ProxyHandler({"https": selected_proxy}))
              if selected_proxy else request.build_opener())
    call = request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "User-Agent": "HT-Voice-Lab/0.1"})
    try:
        payload = _open_json(opener, call, timeout)
    except HTTPError as exc:
        message = _error_message(exc)
        raise OSError("Groq API access check {}: {}{}".format(
            exc.code, message, _permission_hint(exc.code))) from None
    return {item.get("id") for item in payload.get("data", []) if item.get("id")}


def check_model_access(models, timeout=10.0, api_key=None, proxy=None):
    available = available_models(timeout, api_key, proxy)
    missing = sorted(set(models) - available)
    if missing:
        raise OSError("Groq models are unavailable to this project: {}. Check "
                      "https://console.groq.com/settings/limits".format(
                          ", ".join(missing)))
    return sorted(set(models))


def chat_completion(payload, timeout=10.0, api_key=None, proxy=None):
    """Call Groq Chat Completions without exposing credentials to callers."""
    key = api_key or load_api_key()
    if not key:
        raise ValueError("GROQ_API_KEY is not set; configure it locally and do not paste it into chat")
    body = json.dumps(payload, ensure_ascii=False).encode()
    selected_proxy = proxy or os.environ.get("GROQ_HTTPS_PROXY")
    opener = (request.build_opener(request.ProxyHandler({"https": selected_proxy}))
              if selected_proxy else request.build_opener())
    for attempt in range(5):
        call = request.Request(
            "https://api.groq.com/openai/v1/chat/completions", data=body, method="POST",
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json",
                     "User-Agent": "HT-Voice-Lab/0.1"})
        try:
            return _open_json(opener, call, timeout)
        except HTTPError as exc:
            message = _error_message(exc)
            if exc.code == 429 and attempt < 4:
                time.sleep(_retry_after_seconds(exc.headers, str(message), attempt))
                continue
            raise OSError("Groq API {}: {}{}".format(
                exc.code, message, _permission_hint(exc.code))) from None


class GroqSemanticJudge:
    name = "groq-strict-json-v1"

    def __init__(self, model="qwen/qwen3.8-27b", timeout=10.0, api_key=None,
                 transport=None, proxy=None, minimum_interval=0.0):
        self.model = model
        self.timeout = timeout
        if timeout <= 0:
            raise ValueError("Groq timeout must be positive")
        if minimum_interval < 0:
            raise ValueError("Groq minimum interval cannot be negative")
        self.api_key = api_key or load_api_key()
        self.transport = transport or self._post
        self.proxy = proxy or os.environ.get("GROQ_HTTPS_PROXY")
        self.minimum_interval = minimum_interval
        self.last_request_started = None
        if not self.api_key and transport is None:
            raise ValueError("GROQ_API_KEY is not set; configure it locally and do not paste it into chat")

    def _post(self, payload):
        return chat_completion(payload, self.timeout, self.api_key, self.proxy)

    def assess(self, speaker, contribution_ids, text, close_reason, context):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "speaker": speaker, "contributions": context,
                    "latest_text": text, "acoustic_close_reason": close_reason,
                }, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_completion_tokens": 160,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "semantic_turn", "strict": True, "schema": SCHEMA},
            },
        }
        started = time.monotonic()
        if self.last_request_started is not None:
            wait = max(0, self.minimum_interval - (started - self.last_request_started))
            if wait:
                time.sleep(wait)
        request_started = time.monotonic()
        self.last_request_started = request_started
        response = self.transport(payload)
        finished = time.monotonic()
        content = response["choices"][0]["message"]["content"]
        item = json.loads(content)
        if item.get("status") not in {"complete", "incomplete", "uncertain"}:
            raise ValueError("Groq semantic response has an invalid status")
        confidence = float(item["confidence"])
        if not 0 <= confidence <= 1 or not isinstance(item.get("reason"), str):
            raise ValueError("Groq semantic response does not match the expected schema")
        return dict(turn_complete={"complete": True, "incomplete": False,
                                   "uncertain": None}[item["status"]],
                    confidence=confidence, reason=item["reason"],
                    decision_source=self.name, model=self.model,
                    latency_s=finished - started,
                    provider_latency_s=finished - request_started,
                    throttle_wait_s=request_started - started)
