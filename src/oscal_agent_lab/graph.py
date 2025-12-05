# src/oscal_agent_lab/graph.py
"""LangGraph workflow wiring for OSCAL Agent Lab."""

from __future__ import annotations

from langgraph.graph import StateGraph, MessagesState, START, END

from .agents.explainer import explainer_node


def build_graph():
    """
    Build the LangGraph workflow.

    v0.1: Simple linear graph: START -> explainer -> END
    Future: Add routing to diff, profile_builder, validator based on intent.
    """
    graph_builder = StateGraph(MessagesState)

    # Add nodes
    graph_builder.add_node("explainer", explainer_node)

    # Wire edges (simple for v0.1)
    graph_builder.add_edge(START, "explainer")
    graph_builder.add_edge("explainer", END)

    return graph_builder.compile()


def build_multi_agent_graph():
    """
    Build a more sophisticated graph with multiple agents (future).

    TODO: Add:
    - Router node to classify user intent
    - Conditional edges to different agents
    - Loops for clarification
    """
    raise NotImplementedError("Multi-agent graph coming in v0.5")


# Pre-compiled graph for import convenience
graph = build_graph()
