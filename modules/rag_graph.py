"""
LangGraph 编排层：CRAG（Corrective RAG）自纠错检索。

为什么这一层才需要 LangGraph：
- simple 流程是直线调用，条件分支与循环都不存在，用图反而是过度设计
- CRAG 有真正的分支与循环：检索 → 相关性评分 → 不相关则改写查询 → 重新检索
  （最多 MAX_RETRIEVE_RETRIES 次）→ 仍不相关才走兜底回答
- 多轮对话需要跨节点共享状态（问题 / 历史 / 检索结果 / 重试次数）

职责边界：本模块只做编排，检索、重排、生成全部复用现有实现
（HybridRetriever / rerank_chunks / LCEL 链），不重复造轮子。
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from config import MAX_HISTORY_TURNS, MAX_RETRIEVE_RETRIES
from modules.llm_backends import build_chat_model, invoke_chain
from modules.reranker import rerank_chunks

# 相关性评分提示词：让模型只输出 relevant / irrelevant
GRADER_SYSTEM = (
    "你负责判断检索到的资料能否回答用户问题。"
    "只输出一个单词：能回答输出 relevant，不能回答输出 irrelevant。不要输出任何解释。"
)

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GRADER_SYSTEM),
        (
            "human",
            "问题：{question}\n\n检索到的资料：\n{context}\n\n判断：",
        ),
    ]
)

# 查询改写提示词：结合多轮历史把指代补全成可独立检索的问题
REWRITE_SYSTEM = (
    "你是查询改写助手。结合对话历史，把用户的最新问题改写成"
    "一个不依赖上下文、可独立检索的完整问题。只输出改写后的问题本身。"
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM),
        (
            "human",
            "对话历史：\n{history}\n\n最新问题：{question}\n\n改写后的问题：",
        ),
    ]
)


class RagState(TypedDict, total=False):
    """图在节点间流转的状态。"""

    question: str
    history: List[Dict[str, str]]
    retriever: Any
    top_k: int
    chunks: List[Dict[str, Any]]
    relevant: bool
    empty: bool
    retries: int
    answer: str
    fallback: bool


def _format_history(history: List[Dict[str, str]] | None) -> str:
    """
    把对话历史格式化成文本，只保留最近 MAX_HISTORY_TURNS 轮。
    """
    if not history:
        return "（无历史）"

    recent = history[-MAX_HISTORY_TURNS:]
    lines = []

    for turn in recent:
        question = turn.get("question", "")
        answer = turn.get("answer", "")
        if question:
            lines.append(f"用户：{question}")
        if answer:
            lines.append(f"助手：{answer}")

    return "\n".join(lines) if lines else "（无历史）"


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    复用 rag_chain 的上下文拼装逻辑，避免两处实现不一致。
    """
    from modules.rag_chain import build_context

    return build_context(chunks)


def retrieve_node(state: RagState) -> Dict[str, Any]:
    """检索节点：混合检索 + 重排。"""
    retriever = state["retriever"]
    top_k = state.get("top_k", 5)

    candidates = retriever.retrieve(question=state["question"])
    chunks = rerank_chunks(
        query=state["question"],
        retrieved_chunks=candidates,
        top_k=top_k,
    )

    return {"chunks": chunks}


def grade_node(state: RagState) -> Dict[str, Any]:
    """相关性评分节点：判断召回内容能否回答问题。"""
    chunks = state.get("chunks") or []

    # 没有召回任何内容：标记 empty，跳过无意义的改写重试直接兜底
    if not chunks:
        return {"relevant": False, "empty": True}

    chain = GRADER_PROMPT | build_chat_model() | StrOutputParser()

    verdict = invoke_chain(chain,
        {
            "question": state["question"],
            "context": _build_context(chunks),
        }
    )

    return {"relevant": "irrelevant" not in verdict.strip().lower(), "empty": False}


def rewrite_node(state: RagState) -> Dict[str, Any]:
    """查询改写节点：结合历史把问题改写成可独立检索的形式。"""
    chain = REWRITE_PROMPT | build_chat_model() | StrOutputParser()

    rewritten = invoke_chain(chain,
        {
            "history": _format_history(state.get("history")),
            "question": state["question"],
        }
    )

    rewritten = rewritten.strip()

    return {
        "question": rewritten or state["question"],
        "retries": state.get("retries", 0) + 1,
    }


def generate_node(state: RagState) -> Dict[str, Any]:
    """生成节点：复用 rag_chain 的 prompt 与 LCEL 链。"""
    from modules.rag_chain import build_prompt, invoke_llm

    prompt = build_prompt(state["question"], state.get("chunks") or [])

    return {"answer": invoke_llm(prompt)}


def fallback_node(state: RagState) -> Dict[str, Any]:
    """兜底节点：重试耗尽仍不相关时诚实告知，不编造答案。"""
    return {
        "answer": (
            "根据当前检索到的资料，无法完整回答该问题。"
            "已尝试改写查询并重新检索，仍未找到足够相关的课程内容。"
        ),
        "fallback": True,
    }


def route_after_grade(state: RagState) -> str:
    """
    条件边：评分后决定生成、改写重试还是兜底。
    """
    if state.get("relevant"):
        return "generate"

    # 检索完全为空：知识库里没有相关内容，改写多少次都检索不到，
    # 直接兜底，避免无谓的 LLM 调用与等待
    if state.get("empty"):
        return "fallback"

    if state.get("retries", 0) < MAX_RETRIEVE_RETRIES:
        return "rewrite"

    return "fallback"


def build_rag_graph() -> Any:
    """
    组装并编译 CRAG 状态图：

    retrieve → grade →[relevant]→ generate → END
                    →[irrelevant & 重试未耗尽]→ rewrite → retrieve（循环）
                    →[irrelevant & 重试耗尽]→ fallback → END
    """
    graph = StateGraph(RagState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def run_crag(
    question: str,
    retriever: Any,
    top_k: int = 5,
    history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    执行一次 CRAG 流程。

    返回:
        {"question": 最终使用的查询（可能被改写）, "chunks": 召回片段,
         "answer": 答案, "retries": 重试次数, "fallback": 是否兜底}
    """
    graph = build_rag_graph()

    # history 可能来自 pydantic 模型（API 请求体），统一转成 dict
    normalized_history: List[Dict[str, str]] = []
    for turn in history or []:
        if hasattr(turn, "model_dump"):
            turn = turn.model_dump()
        normalized_history.append(
            {
                "question": str(turn.get("question", "")),
                "answer": str(turn.get("answer", "")),
            }
        )

    final_state = graph.invoke(
        {
            "question": question,
            "history": normalized_history,
            "retriever": retriever,
            "top_k": top_k,
            "retries": 0,
        }
    )

    return {
        "question": final_state["question"],
        "original_question": question,
        "chunks": final_state.get("chunks") or [],
        "answer": final_state["answer"],
        "retries": final_state.get("retries", 0),
        "fallback": bool(final_state.get("fallback", False)),
    }
