"""Catch accidental Chinese task inputs while allowing Chinese documentation.

This is a script check for the English benchmark, not a language classifier.
Legacy experiments retain their original inputs and frozen implementations.
"""
import json
import re

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")


def require_english_task(value, context):
    if isinstance(value, str):
        # Chat messages can themselves contain a JSON-encoded public input.
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            require_english_task(decoded, context)
        if _CJK.search(value):
            raise ValueError(f"{context} contains Chinese task text. Use versioned English inputs; "
                             "replay legacy experiments with their frozen implementation.")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_english_task(key, context)
            require_english_task(item, context)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_english_task(item, context)
