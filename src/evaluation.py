from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from src.baselines import SYSTEM_RUNNERS
from src.config import EVAL_DIR, OUTPUTS_DIR, USE_LLM_JUDGE, ensure_dirs
from src.llm_client import get_llm_client
from src.utils import load_json, parse_json_object, write_json


def recall_at_k(evidence: List[Dict[str, Any]], gold_evidence: List[str], k: int = 3) -> int:
    if not gold_evidence:
        return 0
    top_sources = []
    for item in evidence[:k]:
        source = str(item.get("source") or item.get("metadata", {}).get("source") or "").lower()
        top_sources.append(source)
    for gold in gold_evidence:
        gold_lower = str(gold).lower()
        if any(gold_lower in source or source in gold_lower for source in top_sources if source):
            return 1
    return 0


def heuristic_success(answer: str, gold_answer: str) -> int:
    if not gold_answer:
        return 0
    answer_lower = answer.lower()
    keywords = [w.strip(".,;:()[]{}\"'").lower() for w in gold_answer.split() if len(w) >= 5]
    if not keywords:
        return int(gold_answer.lower() in answer_lower)
    matched = sum(1 for w in keywords if w in answer_lower)
    return int(matched >= max(2, len(keywords) // 4))


def heuristic_grounded(answer: str, evidence: List[Dict[str, Any]]) -> float:
    if not evidence or not answer:
        return 0.0
    evidence_text = " ".join([str(item.get("content", "")) for item in evidence]).lower()
    answer_words = [w.strip(".,;:()[]{}\"'").lower() for w in answer.split() if len(w) >= 5]
    if not answer_words:
        return 0.0
    matched = sum(1 for w in answer_words if w in evidence_text)
    return round(min(1.0, matched / max(4, len(answer_words) * 0.35)), 3)


def llm_judge(question: str, gold_answer: str, answer: str, evidence: List[Dict[str, Any]]) -> Tuple[int, float, str]:
    if not USE_LLM_JUDGE:
        return heuristic_success(answer, gold_answer), heuristic_grounded(answer, evidence), "heuristic"

    evidence_text = "\n\n".join(
        [f"[{i+1}] {item.get('source','unknown')} | {item.get('content','')[:900]}" for i, item in enumerate(evidence)]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evaluation judge for a personalised Java programming assistant. "
                "Return strict JSON only: {\"task_success\": 0 or 1, \"groundedness\": number from 0 to 1, \"reason\": \"...\"}. "
                "Task success means the answer is semantically consistent with the gold answer. "
                "Groundedness means the answer is supported by retrieved evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\nGold answer:\n{gold_answer}\n\nRetrieved evidence:\n{evidence_text}\n\nSystem answer:\n{answer}"
            ),
        },
    ]
    try:
        llm = get_llm_client()
        response, _usage = llm.chat(messages, temperature=0.0, max_tokens=260)
        data = parse_json_object(response)
        if data:
            success = int(data.get("task_success", 0))
            groundedness = float(data.get("groundedness", 0.0))
            reason = str(data.get("reason", ""))
            return success, max(0.0, min(1.0, groundedness)), reason
    except Exception as exc:
        return heuristic_success(answer, gold_answer), heuristic_grounded(answer, evidence), f"judge failed, heuristic used: {exc}"
    return heuristic_success(answer, gold_answer), heuristic_grounded(answer, evidence), "judge invalid, heuristic used"


def run_evaluation(questions_path: Path | None = None, systems: List[str] | None = None) -> Dict[str, Any]:
    ensure_dirs()
    questions_path = questions_path or (EVAL_DIR / "benchmark_questions.json")
    questions = load_json(questions_path, default=[])
    if not questions:
        raise RuntimeError(f"No benchmark questions found at {questions_path}")

    selected_systems = systems or list(SYSTEM_RUNNERS.keys())
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []

    # Very simple conversation memory only for follow-up questions.
    chat_history: List[Dict[str, str]] = []

    for q in tqdm(questions, desc="Evaluating questions"):
        qid = q.get("id", "")
        family = q.get("family", "")
        question = q.get("question", "")
        gold_answer = q.get("gold_answer", "")
        gold_evidence = q.get("gold_evidence", []) or []
        if isinstance(gold_evidence, str):
            gold_evidence = [gold_evidence]

        for system_name in selected_systems:
            runner = SYSTEM_RUNNERS[system_name]
            result = runner(question, chat_history=chat_history)
            evidence = result.get("evidence", [])
            answer = result.get("answer", "")
            success, groundedness, judge_reason = llm_judge(question, gold_answer, answer, evidence)
            rec3 = recall_at_k(evidence, gold_evidence, k=3)

            row = {
                "question_id": qid,
                "family": family,
                "system": system_name,
                "question": question,
                "answer": answer,
                "gold_answer": gold_answer,
                "gold_evidence": json.dumps(gold_evidence, ensure_ascii=False),
                "retrieved_sources": json.dumps(result.get("retrieved_sources", []), ensure_ascii=False),
                "recall_at_3": rec3,
                "task_success": success,
                "groundedness": groundedness,
                "latency": round(float(result.get("latency", 0.0)), 4),
                "token_usage": int(result.get("token_usage", 0) or 0),
                "tool_calls": int(result.get("tool_calls", 0) or 0),
                "query_type": result.get("query_type", ""),
                "retrieval_plan": result.get("retrieval_plan", ""),
                "judge_reason": judge_reason,
            }
            rows.append(row)
            if result.get("trace"):
                traces.append({"question_id": qid, "system": system_name, "trace": result.get("trace")})

        # Update memory with a concise gold-oriented context after all systems run.
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": gold_answer or "[benchmark answer hidden]"})
        chat_history = chat_history[-6:]

    results_path = OUTPUTS_DIR / "eval_results.csv"
    summary_path = OUTPUTS_DIR / "eval_summary.json"
    traces_path = OUTPUTS_DIR / "agent_traces.jsonl"

    fieldnames = list(rows[0].keys())
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with traces_path.open("w", encoding="utf-8") as f:
        for item in traces:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["system"]].append(row)

    for system_name, system_rows in grouped.items():
        n = max(1, len(system_rows))
        by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in system_rows:
            by_family[row["family"]].append(row)
        summary[system_name] = {
            "num_questions": n,
            "recall_at_3": round(sum(float(r["recall_at_3"]) for r in system_rows) / n, 3),
            "task_success": round(sum(float(r["task_success"]) for r in system_rows) / n, 3),
            "groundedness": round(sum(float(r["groundedness"]) for r in system_rows) / n, 3),
            "avg_latency": round(sum(float(r["latency"]) for r in system_rows) / n, 3),
            "avg_token_usage": round(sum(float(r["token_usage"]) for r in system_rows) / n, 1),
            "avg_tool_calls": round(sum(float(r["tool_calls"]) for r in system_rows) / n, 2),
            "by_family": {
                fam: {
                    "task_success": round(sum(float(r["task_success"]) for r in fam_rows) / len(fam_rows), 3),
                    "groundedness": round(sum(float(r["groundedness"]) for r in fam_rows) / len(fam_rows), 3),
                    "recall_at_3": round(sum(float(r["recall_at_3"]) for r in fam_rows) / len(fam_rows), 3),
                }
                for fam, fam_rows in by_family.items()
            },
        }

    write_json(summary_path, summary)
    print(f"Evaluation completed.\n- Results: {results_path}\n- Summary: {summary_path}\n- Traces: {traces_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and final-agent evaluation.")
    parser.add_argument("--questions", type=str, default=str(EVAL_DIR / "benchmark_questions.json"))
    parser.add_argument(
        "--systems",
        nargs="*",
        default=list(SYSTEM_RUNNERS.keys()),
        choices=list(SYSTEM_RUNNERS.keys()),
        help="Systems to evaluate.",
    )
    args = parser.parse_args()
    run_evaluation(Path(args.questions), systems=args.systems)


if __name__ == "__main__":
    main()
