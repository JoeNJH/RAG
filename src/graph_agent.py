from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.config import MAX_REPAIR_ATTEMPTS, USE_LLM_ROUTER, USE_LLM_VERIFIER
from src.llm_client import get_llm_client
from src.retriever import format_evidence, get_retriever
from src.utils import parse_json_object

QueryType = Literal["factual", "cross_modal", "analytical", "follow_up", "debugging"]
RetrievalPlan = Literal["text", "image", "hybrid"]


class AgentState(TypedDict, total=False):
    question: str
    chat_history: List[Dict[str, str]]
    query_type: str
    retrieval_plan: str
    evidence: List[Dict[str, Any]]
    evidence_text: str
    answer: str
    grounded: bool
    verification_reason: str
    attempts: int
    token_usage: int
    tool_calls: int
    trace: List[Dict[str, Any]]


def _add_trace(state: AgentState, node: str, payload: Dict[str, Any]) -> None:
    state.setdefault("trace", []).append({"node": node, **payload})


def _add_tokens(state: AgentState, usage: Dict[str, int]) -> None:
    state["token_usage"] = state.get("token_usage", 0) + int(usage.get("total_tokens", 0) or 0)


def rule_based_query_type(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> QueryType:
    q = question.lower()
    if chat_history and any(word in q for word in ["next", "previous", "刚才", "上一个", "继续", "based on my previous"]):
        return "follow_up"
    if any(word in q for word in ["screenshot", "image", "diagram", "figure", "chart", "uml", "架构图", "截图", "图片", "图"]):
        return "cross_modal"
    if any(word in q for word in ["error", "exception", "bug", "debug", "failed", "failure", "报错", "异常", "错误", "排查"]):
        return "debugging"
    if any(word in q for word in ["why", "compare", "difference", "explain", "how", "trade-off", "分析", "比较", "为什么", "如何"]):
        return "analytical"
    return "factual"


def classify_query_node(state: AgentState) -> AgentState:
    question = state["question"]
    chat_history = state.get("chat_history", [])
    query_type = rule_based_query_type(question, chat_history)

    if USE_LLM_ROUTER:
        llm = get_llm_client()
        history_hint = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in chat_history[-4:]])
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify a user's Java programming assistant question. Return strict JSON only. "
                    "Allowed query_type values: factual, cross_modal, analytical, follow_up, debugging."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nRecent chat history:\n{history_hint}\n\n"
                    "Return JSON: {\"query_type\": \"...\", \"reason\": \"...\"}"
                ),
            },
        ]
        try:
            response, usage = llm.chat(messages, temperature=0.0, max_tokens=180)
            _add_tokens(state, usage)
            data = parse_json_object(response)
            if data and data.get("query_type") in {"factual", "cross_modal", "analytical", "follow_up", "debugging"}:
                query_type = data["query_type"]  # type: ignore[assignment]
        except Exception:
            # Rule-based fallback is deterministic and keeps evaluation running.
            pass

    state["query_type"] = query_type
    state["tool_calls"] = state.get("tool_calls", 0) + 1
    _add_trace(state, "classify_query", {"query_type": query_type})
    return state


def plan_retrieval_node(state: AgentState) -> AgentState:
    query_type = state.get("query_type", "factual")
    if query_type == "cross_modal":
        plan: RetrievalPlan = "image"
    elif query_type in {"analytical", "follow_up", "debugging"}:
        plan = "hybrid"
    else:
        plan = "text"
    state["retrieval_plan"] = plan
    state["tool_calls"] = state.get("tool_calls", 0) + 1
    _add_trace(state, "plan_retrieval", {"retrieval_plan": plan})
    return state


def retrieve_evidence_node(state: AgentState) -> AgentState:
    question = state["question"]
    query_type = state.get("query_type", "factual")
    plan = state.get("retrieval_plan", "hybrid")

    # Follow-up questions need recent context added to retrieval query.
    if query_type == "follow_up" and state.get("chat_history"):
        recent = " ".join([m.get("content", "") for m in state.get("chat_history", [])[-4:]])
        retrieval_query = f"{recent}\nCurrent question: {question}"
    else:
        retrieval_query = question

    retriever = get_retriever()
    evidence = retriever.retrieve(retrieval_query, mode=plan)  # type: ignore[arg-type]
    state["evidence"] = evidence
    state["evidence_text"] = format_evidence(evidence)
    state["tool_calls"] = state.get("tool_calls", 0) + 1
    _add_trace(
        state,
        "retrieve_evidence",
        {
            "retrieval_plan": plan,
            "num_evidence": len(evidence),
            "sources": [item.get("source") for item in evidence],
        },
    )
    return state


def generate_answer_node(state: AgentState) -> AgentState:
    llm = get_llm_client()
    question = state["question"]
    evidence_text = state.get("evidence_text", "No retrieved evidence.")
    history = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in state.get("chat_history", [])[-4:]])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a personalised Java programming assistant. Answer using only the retrieved evidence. "
                "If evidence is insufficient, say this clearly. Cite evidence numbers like [1], [2]. "
                "Be practical and concise. For debugging questions, give step-by-step checks grounded in evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Recent chat history:\n{history}\n\n"
                f"Question:\n{question}\n\n"
                f"Retrieved evidence:\n{evidence_text}\n\n"
                "Now answer the question with evidence citations."
            ),
        },
    ]
    answer, usage = llm.chat(messages)
    _add_tokens(state, usage)
    state["answer"] = answer
    state["tool_calls"] = state.get("tool_calls", 0) + 1
    _add_trace(state, "generate_answer", {"answer_preview": answer[:200]})
    return state


def verify_answer_node(state: AgentState) -> AgentState:
    answer = state.get("answer", "")
    evidence_text = state.get("evidence_text", "")
    grounded = bool(answer and state.get("evidence"))
    reason = "Rule-based check: answer and retrieved evidence are present."

    if USE_LLM_VERIFIER:
        llm = get_llm_client()
        messages = [
            {
                "role": "system",
                "content": (
                    "You verify whether an answer is supported by retrieved evidence. Return strict JSON only: "
                    "{\"grounded\": true/false, \"reason\": \"...\"}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state.get('question','')}\n\n"
                    f"Evidence:\n{evidence_text}\n\n"
                    f"Answer:\n{answer}\n\n"
                    "Is the answer grounded in the evidence?"
                ),
            },
        ]
        try:
            response, usage = llm.chat(messages, temperature=0.0, max_tokens=220)
            _add_tokens(state, usage)
            data = parse_json_object(response)
            if data is not None:
                grounded = bool(data.get("grounded", grounded))
                reason = str(data.get("reason", reason))
        except Exception as exc:
            reason = f"Verifier failed; used rule-based fallback. Error: {exc}"

    state["grounded"] = grounded
    state["verification_reason"] = reason
    state["tool_calls"] = state.get("tool_calls", 0) + 1
    _add_trace(state, "verify_answer", {"grounded": grounded, "reason": reason})
    return state


def repair_answer_node(state: AgentState) -> AgentState:
    attempts = state.get("attempts", 0) + 1
    state["attempts"] = attempts
    state["retrieval_plan"] = "hybrid"
    # Make the query more conservative and evidence-seeking in the next retrieval round.
    state["question"] = state["question"] + "\nFind direct supporting evidence from both Java text notes and image-derived chunks."
    state["tool_calls"] = state.get("tool_calls", 0) + 1
    _add_trace(state, "repair_answer", {"attempts": attempts, "new_retrieval_plan": "hybrid"})
    return state


def should_repair(state: AgentState) -> str:
    if state.get("grounded", False):
        return "finish"
    if state.get("attempts", 0) >= MAX_REPAIR_ATTEMPTS:
        return "finish"
    return "repair"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_query", classify_query_node)
    graph.add_node("plan_retrieval", plan_retrieval_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("verify_answer", verify_answer_node)
    graph.add_node("repair_answer", repair_answer_node)

    graph.set_entry_point("classify_query")
    graph.add_edge("classify_query", "plan_retrieval")
    graph.add_edge("plan_retrieval", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_answer")
    graph.add_edge("generate_answer", "verify_answer")
    graph.add_conditional_edges("verify_answer", should_repair, {"repair": "repair_answer", "finish": END})
    graph.add_edge("repair_answer", "retrieve_evidence")
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    initial_state: AgentState = {
        "question": question,
        "chat_history": chat_history or [],
        "attempts": 0,
        "token_usage": 0,
        "tool_calls": 0,
        "trace": [],
    }
    result = get_graph().invoke(initial_state)
    return dict(result)
