"""Small synchronous client for a configured Chat Completions endpoint."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from task_undo.common.task_language import require_english_task


def validate_config(config):
    expected = {"base_url", "model", "api_key_env", "max_turns", "max_actions", "max_tokens", "request_timeout_s"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Use exactly the documented model settings; put credentials in the environment")
    if not all(isinstance(config[k], str) for k in ("base_url", "model", "api_key_env")):
        raise ValueError("Model endpoint, name and key variable must be strings")
    url = urllib.parse.urlsplit(config["base_url"])
    if (url.scheme not in ("http", "https") or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
    if not config["model"] or not config["api_key_env"]:
        raise ValueError("Model name and API key environment variable name are required")
    for key in ("max_turns", "max_actions", "max_tokens", "request_timeout_s"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")


class ChatModel:
    def __init__(self, config):
        validate_config(config)
        self.config = config
        # Never inherit shell proxy settings, including for local services.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def complete(self, messages, *, response_schema=None):
        require_english_task(messages, "Model request")
        require_english_task(response_schema, "Response schema")
        payload = {"model": self.config["model"], "messages": messages,
                   "max_tokens": self.config["max_tokens"], "temperature": 0,
                   "response_format": {"type": "json_object"}}
        if response_schema is not None:
            if not isinstance(response_schema, dict):
                raise ValueError("Response schema must be a JSON object")
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "task_undo_decisions", "strict": True, "schema": response_schema}}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config["api_key_env"])
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self.config["base_url"].rstrip("/") + "/chat/completions",
                                         data=json.dumps(payload).encode(), headers=headers)
        try:
            with self.opener.open(request, timeout=self.config["request_timeout_s"]) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Avoid logging provider bodies which may echo credentials.
            raise RuntimeError(f"Model HTTP error {exc.code}") from None
        except (OSError, ValueError):
            raise RuntimeError("Model connection failed or returned invalid JSON") from None
        try:
            choice = result["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Model response incomplete or refused")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Model returned no text")
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("Model returned an incomplete or unsupported response") from None
        return {"content": content, "model": result.get("model"), "id": result.get("id"),
                "usage": result.get("usage"), "finish_reason": choice["finish_reason"]}
