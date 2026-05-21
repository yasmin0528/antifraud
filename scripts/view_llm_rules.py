#!/usr/bin/env python3
"""
查看 LLM 生成的规则。

用法：
    python scripts/view_llm_rules.py                              # 使用默认路径 llm_rules.json
    python scripts/view_llm_rules.py --file outputs/llm_rules.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="查看 LLM 生成的规则")
    parser.add_argument("--file", type=str, default="llm_rules.json", help="规则 JSON 文件路径")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        print("\n训练过程中 LLM 更新规则时会自动保存到 llm_rules.json。")
        print("如果没有这个文件，说明 LLM 未连接（fallback 模式不保存规则）。")
        return

    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    if not rules:
        print("规则文件为空。")
        return

    is_fallback = any(r.get("rule", "").startswith("DEFAULT_") for r in rules)

    print(f"\n共 {len(rules)} 条规则")
    if is_fallback:
        print("(当前为 fallback 默认规则，非 LLM 生成)")
    print("=" * 60)

    for i, rule in enumerate(rules, 1):
        print(f"\n--- 规则 {i} ---")
        print(f"  IF:   {rule.get('rule', 'N/A')}")
        print(f"  THEN: {rule.get('label', 'suspicious')}")
        print(f"  置信度: {rule.get('confidence', 'N/A')}")

    print("=" * 60)


if __name__ == "__main__":
    main()
