# src/oscal_agent_lab/agents/explainer.py
"""ExplainerAgent: RAG-based Q&A over OSCAL content."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, HumanMessage

from ..config import OPENAI_MODEL
from ..vectorstore import get_retriever

# Lazy initialization to avoid loading at import time
_llm: ChatOpenAI | None = None
_retriever = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    return _llm


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever(k=6)
    return _retriever


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an OSCAL cybersecurity and compliance copilot.

You answer questions about NIST SP 800-53 controls and related OSCAL content.

Guidelines:
- Use only the provided OSCAL snippets as ground truth
- Refer to controls by their IDs (e.g., AC-2, SI-4)
- Explain technical terms when helpful
- If the snippets don't contain enough information, say so clearly
- For implementation questions, suggest practical approaches

You help security teams understand control requirements, plan implementations,
and navigate the OSCAL ecosystem.""",
        ),
        (
            "human",
            """User question:
{question}

Relevant OSCAL content:
{context}

Provide a clear, helpful answer based on the OSCAL content above.""",
        ),
    ]
)


def explainer_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: uses RAG over OSCAL catalog to answer the latest user question.

    Expects a MessagesState-like dict with `messages` list.
    Returns dict with new messages to add to state.
    """
    messages: list[AnyMessage] = state.get("messages", [])
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, HumanMessage):
        # Only respond when the last message is from the user
        return {}

    question = last.content

    # Retrieve relevant OSCAL documents
    retriever = _get_retriever()
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Generate response
    llm = _get_llm()
    prompt_messages = _PROMPT.format_messages(question=question, context=context)
    result = llm.invoke(prompt_messages)

    # LangGraph will merge this into state (using add_messages reducer)
    return {"messages": [result]}


# For direct testing without the graph
def explain(question: str) -> str:
    """
    Direct function to ask the explainer a question.

    Useful for testing outside of the LangGraph context.
    """
    retriever = _get_retriever()
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    llm = _get_llm()
    prompt_messages = _PROMPT.format_messages(question=question, context=context)
    result = llm.invoke(prompt_messages)

    return result.content
