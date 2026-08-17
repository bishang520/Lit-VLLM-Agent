"""启动 API 服务（真实 vLLM 或 mock 联调）。"""

from __future__ import annotations

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.serving.app import main

if __name__ == "__main__":
    sys.exit(main())
