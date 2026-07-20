#!/usr/bin/env python3
"""校验 Conventional Commits 格式的提交信息。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

COMMIT_MSG_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else None

PATTERN = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\((?P<scope>[^)]+)\))?"
    r": (?P<subject>.+)$"
)

MAX_SUBJECT_LEN = 72


def main() -> int:
    if COMMIT_MSG_FILE is None or not COMMIT_MSG_FILE.exists():
        return 0

    lines = COMMIT_MSG_FILE.read_text(encoding="utf-8").splitlines()
    if not lines:
        print("错误：提交信息为空", file=sys.stderr)
        return 1

    subject_line = lines[0].strip()
    if subject_line.startswith("#") or subject_line.startswith("Merge "):
        return 0

    match = PATTERN.match(subject_line)
    if not match:
        print(
            "错误：提交信息不符合规范\n"
            "格式：<type>(<scope>): <subject>\n"
            "示例：feat(tensor): 添加 GPU 性能对比脚本",
            file=sys.stderr,
        )
        return 1

    subject = match.group("subject")
    if len(subject) > MAX_SUBJECT_LEN:
        print(
            f"错误：subject 超过 {MAX_SUBJECT_LEN} 字符（当前 {len(subject)}）",
            file=sys.stderr,
        )
        return 1

    if subject.endswith("."):
        print("错误：subject 末尾不应有句号", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
