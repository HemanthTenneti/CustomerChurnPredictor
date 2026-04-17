"""LangGraph-based churn retention agent — 5-node state machine with RAG and Groq LLM."""

import os
import sys
import logging
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END

from agent.prompts import EXPLANATION_PROMPT, RECOMMENDATION_PROMPT, SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema flowing through the LangGraph agent pipeline."""

    customer_features: dict
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    risk_factors: list[str]
    retrieved_context: str
    explanation: str
    recommendations: list[str]
    executive_summary: str
    error: str | None


def get_llm():
    """Return a ChatGroq LLM instance. Single point to swap models."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Node functions ───────────────────────────────────────────────────────────


def _predict(state: AgentState) -> dict:
    """Run ML model and populate probability, prediction, risk_level, risk_factors."""
    from agent.tools import predict_churn_tool, identify_top_risk_factors_tool

    features = state["customer_features"]
    result = predict_churn_tool.invoke({"customer_features": features})
    factors = identify_top_risk_factors_tool.invoke({"customer_features": features})

    return {
        "churn_probability": result["probability"],
        "churn_prediction": result["prediction"],
        "risk_level": result["risk_level"],
        "risk_factors": factors,
    }


def _retrieve_context(state: AgentState) -> dict:
    """Query RAG retriever with risk profile to get relevant strategies."""
    from agent.tools import retrieve_retention_strategies_tool

    query = (
        f"retention strategies for {state['risk_level']} risk: "
        f"{', '.join(state['risk_factors'])}"
    )
    context = retrieve_retention_strategies_tool.invoke({"query": query})
    return {"retrieved_context": context}


def _explain(state: AgentState) -> dict:
    """Call Groq LLM to explain why the customer is at risk."""
    llm = get_llm()
    chain = EXPLANATION_PROMPT | llm
    response = chain.invoke(
        {
            "customer_features": _fmt_features(state["customer_features"]),
            "churn_probability": f"{state['churn_probability']:.1%}",
            "risk_level": state["risk_level"],
            "risk_factors": ", ".join(state["risk_factors"]),
            "retrieved_context": state["retrieved_context"],
        }
    )
    return {"explanation": response.content.strip()}


def _recommend(state: AgentState) -> dict:
    """Call Groq LLM to generate 3 personalized retention recommendations."""
    llm = get_llm()
    chain = RECOMMENDATION_PROMPT | llm
    response = chain.invoke(
        {
            "customer_features": _fmt_features(state["customer_features"]),
            "churn_probability": f"{state['churn_probability']:.1%}",
            "risk_level": state["risk_level"],
            "explanation": state["explanation"],
            "retrieved_context": state["retrieved_context"],
        }
    )
    # Parse numbered list into clean list of strings
    recs = [
        line.lstrip("0123456789.-) ")
        for line in response.content.strip().split("\n")
        if line.strip()
    ]
    return {"recommendations": recs[:3]}


def _format_output(state: AgentState) -> dict:
    """Call Groq LLM for a one-sentence executive summary."""
    llm = get_llm()
    chain = SUMMARY_PROMPT | llm
    response = chain.invoke(
        {
            "customer_features": _fmt_features(state["customer_features"]),
            "churn_probability": f"{state['churn_probability']:.1%}",
            "risk_level": state["risk_level"],
            "recommendations": " | ".join(state.get("recommendations", [])),
        }
    )
    return {"executive_summary": response.content.strip()}


def _fmt_features(features: dict) -> str:
    """Format customer features dict into a readable multi-line string."""
    return "\n".join(f"  {k}: {v}" for k, v in features.items())


# ── Error-safe node wrappers ─────────────────────────────────────────────────


def _safe(node_fn, name: str):
    """Wrap a node function with error handling so the graph never crashes."""

    def wrapper(state: AgentState) -> dict:
        try:
            return node_fn(state)
        except Exception as exc:
            logger.exception(f"Agent node '{name}' failed")
            return {"error": f"Agent error in {name}: {exc}"}

    wrapper.__name__ = name
    return wrapper


# ── Graph construction ───────────────────────────────────────────────────────


def _build_graph() -> StateGraph:
    """Construct the 5-node LangGraph StateGraph."""
    g = StateGraph(AgentState)

    g.add_node("predict", _safe(_predict, "predict"))
    g.add_node("retrieve_context", _safe(_retrieve_context, "retrieve_context"))
    g.add_node("explain", _safe(_explain, "explain"))
    g.add_node("recommend", _safe(_recommend, "recommend"))
    g.add_node("format_output", _safe(_format_output, "format_output"))

    g.set_entry_point("predict")
    g.add_edge("predict", "retrieve_context")
    g.add_edge("retrieve_context", "explain")
    g.add_edge("explain", "recommend")
    g.add_edge("recommend", "format_output")
    g.add_edge("format_output", END)

    return g.compile()


# ── Public API ───────────────────────────────────────────────────────────────


class ChurnRetentionAgent:
    """Single public interface to the churn retention LangGraph agent."""

    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self, customer_features: dict) -> AgentState:
        """Execute the full agent pipeline and return the final state."""
        initial: AgentState = {
            "customer_features": customer_features,
            "churn_probability": 0.0,
            "churn_prediction": False,
            "risk_level": "",
            "risk_factors": [],
            "retrieved_context": "",
            "explanation": "",
            "recommendations": [],
            "executive_summary": "",
            "error": None,
        }
        result = self._graph.invoke(initial)
        return result


_agent_instance: ChurnRetentionAgent | None = None


def get_agent() -> ChurnRetentionAgent:
    """Lazy singleton — only one agent instance per process."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ChurnRetentionAgent()
    return _agent_instance


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    profile = {
        "senior": "No",
        "tenure": 2,
        "monthly": 90.0,
        "total": 180.0,
        "gender": "Female",
        "partner": "No",
        "dependents": "No",
        "phone": "Yes",
        "multilines": "No",
        "internet": "Fiber optic",
        "online_sec": "No",
        "online_bkp": "No",
        "device_prot": "No",
        "tech_sup": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless": "Yes",
        "payment": "Electronic check",
    }

    agent = ChurnRetentionAgent()
    state = agent.run(profile)

    print("\n=== Agent Output ===")
    for key, value in state.items():
        print(f"\n--- {key} ---")
        if isinstance(value, list):
            for item in value:
                print(f"  • {item}")
        else:
            print(f"  {value}")
