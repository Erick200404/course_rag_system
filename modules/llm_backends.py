"""
LLM 调用层：使用 LangChain chat model + LCEL 组装生成链。

    提示词模板 -> chat model -> 字符串输出解析器

为什么改用 LCEL：
- 旧实现手写 OpenAI SDK 调用与 Ollama 的 requests 调用，两套代码各自拼接参数
- LCEL 把「提示词 / 模型 / 输出解析」拆成可组合的环节，两个后端只差一个 model 对象
- 后续要加流式输出、结构化输出（JSON）、重试都只需替换链中的某个环节

注意：Ollama 后端由原来的 /api/generate 改为 LangChain ChatOllama 的 /api/chat，
system / user 角色通过 ChatPromptTemplate 正确传递，参数（temperature、num_ctx）保持一致。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    CHAT_MODEL,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)

# 系统提示词：与旧实现保持一致
SYSTEM_PROMPT = "你是一个课程资料问答助手，擅长根据给定资料回答问题。"

# LCEL 链：输入为 {"prompt": ...}
_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{prompt}"),
    ]
)


def build_chat_model() -> BaseChatModel:
    """
    根据配置构造 LangChain chat model。

    返回:
        ChatOpenAI（api 后端）或 ChatOllama（ollama 后端）
    """
    if LLM_BACKEND == "api":
        import os

        from langchain_openai import ChatOpenAI

        api_key = os.getenv("ZHI_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 ZHI_API_KEY，无法使用 API LLM 后端")

        return ChatOpenAI(
            model=CHAT_MODEL,
            api_key=api_key,
            base_url=os.getenv("ZHI_BASE_URL"),
            temperature=0.2,
        )

    if LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            # 与旧实现一致：限制上下文窗口，降低本地 7B 模型的压力
            num_ctx=2048,
            # ChatOllama 没有 timeout 字段，超时通过 httpx client kwargs 传递
            client_kwargs={"timeout": OLLAMA_TIMEOUT},
        )

    raise ValueError(f"不支持的 LLM_BACKEND: {LLM_BACKEND}")


def build_answer_chain(chat_model: BaseChatModel | None = None) -> Any:
    """
    组装 LCEL 生成链：prompt | model | StrOutputParser。
    """
    model = chat_model if chat_model is not None else build_chat_model()

    return _PROMPT_TEMPLATE | model | StrOutputParser()


def invoke_llm(prompt: str, chat_model: BaseChatModel | None = None) -> str:
    """
    调用大模型并返回纯文本答案。

    参数:
        prompt: 完整提示词（资料上下文 + 问题 + 回答要求）
        chat_model: 可选，注入自定义模型（测试用）

    返回:
        模型生成的文本
    """
    chain = build_answer_chain(chat_model)

    return chain.invoke({"prompt": prompt})
