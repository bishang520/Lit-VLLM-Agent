"""JSON Schema 约束解码：schema 定义 + 兜底校验。"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import PROJECT_ROOT

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs" / "serving" / "answer_schema.json"


def load_response_schema(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_SCHEMA_PATH
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def extract_json(text: str) -> dict | None:
    """从模型输出中提取第一个合法的 JSON 对象。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def validate_json(obj: dict, schema: dict) -> tuple[bool, str]:
    """用 jsonschema 校验对象；未安装时仅做结构检查。"""
    try:
        from jsonschema import ValidationError, validate
    except ImportError:
        required = set(schema.get("required", []))
        missing = required - set(obj.keys())
        return (not missing), f"缺少字段: {sorted(missing)}"

    try:
        validate(instance=obj, schema=schema)
        return True, ""
    except ValidationError as e:
        return False, str(e)
