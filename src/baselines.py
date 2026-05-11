from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.graph_agent import ask as final_agent_ask
from src.llm_client import get_llm_client
from src.retriever import format_evidence, get_retriever


def _build_grounded_prompt(question: str, evidence_text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a Java programming assistant. Answer using only the retrieved evidence. "
                "Cite evidence numbers like [1]. If evidence is insufficient, say so."
            ),
        },
        {"role": "user", "content": f"Question:\n{question}\n\nEvidence:\n{evidence_text}\n\nAnswer:"},
    ]


def run_plain_llm(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    llm = get_llm_client()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Java programming assistant. Answer from general knowledge. "
                "Do not claim access to the user's private notes unless provided."
            ),
        },
        {"role": "user", "content": question},
    ]
    answer, usage = llm.chat(messages)
    return {
        "system": "B0_plain_llm",
        "answer": answer,
        "evidence": [],
        "retrieved_sources": [],
        "latency": time.perf_counter() - start,
        "token_usage": usage.get("total_tokens", 0),
        "tool_calls": 1,
        "trace": [],
    }


def run_text_only_rag(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    retriever = get_retriever()
    evidence = retriever.retrieve(question, mode="text")
    evidence_text = format_evidence(evidence)
    llm = get_llm_client()
    answer, usage = llm.chat(_build_grounded_prompt(question, evidence_text))
    return {
        "system": "B1_text_only_rag",
        "answer": answer,
        "evidence": evidence,
        "retrieved_sources": [e.get("source") for e in evidence],
        "latency": time.perf_counter() - start,
        "token_usage": usage.get("total_tokens", 0),
        "tool_calls": 2,
        "trace": [],
    }


def run_multimodal_rag(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    retriever = get_retriever()
    evidence = retriever.retrieve(question, mode="hybrid")
    evidence_text = format_evidence(evidence)
    llm = get_llm_client()
    answer, usage = llm.chat(_build_grounded_prompt(question, evidence_text))
    return {
        "system": "B2_multimodal_rag",
        "answer": answer,
        "evidence": evidence,
        "retrieved_sources": [e.get("source") for e in evidence],
        "latency": time.perf_counter() - start,
        "token_usage": usage.get("total_tokens", 0),
        "tool_calls": 2,
        "trace": [],
    }


def run_final_agent(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    result = final_agent_ask(question, chat_history=chat_history or [])
    evidence = result.get("evidence", [])
    return {
        "system": "B3_final_langgraph_agent",
        "answer": result.get("answer", ""),
        "evidence": evidence,
        "retrieved_sources": [e.get("source") for e in evidence],
        "latency": time.perf_counter() - start,
        "token_usage": result.get("token_usage", 0),
        "tool_calls": result.get("tool_calls", 0),
        "trace": result.get("trace", []),
        "query_type": result.get("query_type", ""),
        "retrieval_plan": result.get("retrieval_plan", ""),
        "grounded_by_verifier": result.get("grounded", False),
        "verification_reason": result.get("verification_reason", ""),
    }


SYSTEM_RUNNERS = {
    "B0_plain_llm": run_plain_llm,
    "B1_text_only_rag": run_text_only_rag,
    "B2_multimodal_rag": run_multimodal_rag,
    "B3_final_langgraph_agent": run_final_agent,
}
