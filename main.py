from __future__ import annotations

import argparse
import json
from typing import Dict, List

from src.baselines import SYSTEM_RUNNERS
from src.graph_agent import ask as agent_ask


def print_agent_result(result: Dict) -> None:
    print("\n=== Final Answer ===")
    print(result.get("answer", ""))
    print("\n=== Agent Metadata ===")
    print(f"query_type: {result.get('query_type', '')}")
    print(f"retrieval_plan: {result.get('retrieval_plan', '')}")
    print(f"grounded: {result.get('grounded', '')}")
    print(f"token_usage: {result.get('token_usage', 0)}")
    print(f"tool_calls: {result.get('tool_calls', 0)}")
    print("\n=== Retrieved Evidence ===")
    for i, ev in enumerate(result.get("evidence", []), start=1):
        meta = ev.get("metadata", {}) or {}
        print(f"[{i}] {meta.get('source', ev.get('source', 'unknown'))} | {meta.get('modality','')} | score={ev.get('score',0):.3f}")


def interactive() -> None:
    chat_history: List[Dict[str, str]] = []
    print("Personalised Java Programming Assistant. Type 'exit' to quit.\n")
    while True:
        question = input("User: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        result = agent_ask(question, chat_history=chat_history)
        print("Assistant:", result.get("answer", ""), "\n")
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": result.get("answer", "")})
        chat_history = chat_history[-6:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the A3 Java Programming Assistant.")
    parser.add_argument("--question", type=str, default="", help="Single question to ask.")
    parser.add_argument(
        "--system",
        type=str,
        default="agent",
        choices=["agent", *SYSTEM_RUNNERS.keys()],
        help="Run final agent or one baseline system.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON result.")
    args = parser.parse_args()

    if args.question:
        if args.system == "agent":
            result = agent_ask(args.question)
        else:
            result = SYSTEM_RUNNERS[args.system](args.question)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print_agent_result(result)
        return

    interactive()


if __name__ == "__main__":
    main()
