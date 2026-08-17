"""Windows 控制台编码适配：避免 GBK 控制台打印中文/特殊符号报错。"""

from __future__ import annotations

import sys


def fix_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
