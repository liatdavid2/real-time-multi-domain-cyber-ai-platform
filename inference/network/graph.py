from functools import lru_cache
from network.features import build_features_from_json
from network.model import load_model
from network.rules import evaluate_rules
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
import os
import pandas as pd
import shap
from typing import Any, Dict, List, TypedDict, Optional

# Load .env from parent folder (inference/)
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

explainer = None
embed_model = None
pinecone_index = None
rag_graph = None


class RAGState(TypedDict, total=False):
    query: str
    namespace: str
    decision: str
    shap_features: List[Dict[str, Any]]
    retrieved_context: List[Dict[str, Any]]
    rag_analysis: Dict[str, Any]



@lru_cache(maxsize=1000)
def cached_embed(query: str):
    return get_embed_model().encode(query).tolist()




def get_embed_model() -> SentenceTransformer:
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embed_model


def get_pinecone_index():
    global pinecone_index

    if pinecone_index is not None:
        return pinecone_index

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "security-knowledge")

    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY environment variable")

    pc = Pinecone(api_key=api_key)
    pinecone_index = pc.Index(index_name)
    return pinecone_index


def extract_proba(model, df):
    # Direct model
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        return probs[1] if len(probs) > 1 else probs[0]

    # sklearn Pipeline
    if hasattr(model, "steps"):
        last_model = model.steps[-1][1]
        if hasattr(last_model, "predict_proba"):
            probs = last_model.predict_proba(df)[0]
            return probs[1] if len(probs) > 1 else probs[0]

    # Fallback
    return None


def unwrap_model_for_shap(model):
    # If model is a Pipeline, extract the final estimator step
    if hasattr(model, "steps"):
        model = model.steps[-1][1]

    # If model is CalibratedClassifierCV, extract the underlying base estimator
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator

    # Fallback: some models expose base_estimator directly
    if hasattr(model, "base_estimator"):
        return model.base_estimator

    # Return model as-is if no wrapping is detected
    return model


def build_query(data: Dict[str, Any], shap_features: List[Dict[str, Any]]) -> str:
    shap_text = ", ".join([
        f"{f['feature']}={round(f['value'], 2)} ({f['direction']})"
        for f in shap_features[:5]
    ])

    return f"""
    Network anomaly detection.

    Flow statistics:
    duration={data.get('dur', 0)},
    sbytes={data.get('sbytes', 0)},
    dbytes={data.get('dbytes', 0)},
    spkts={data.get('spkts', 0)},
    dpkts={data.get('dpkts', 0)},
    sttl={data.get('sttl', 0)},
    dttl={data.get('dttl', 0)},
    sload={data.get('sload', 0)},
    dload={data.get('dload', 0)},
    tcprtt={data.get('tcprtt', 0)}

    Top anomaly indicators:
    {shap_text}
    """


def retrieve_context(query: str, namespace: str = "network-flows", k: int = 3) -> List[Dict[str, Any]]:
    model = get_embed_model()
    index = get_pinecone_index()

    embedding = cached_embed(query)

    result = index.query(
        vector=embedding,
        top_k=k,
        namespace=namespace,
        include_metadata=True
    )

    matches = result.get("matches", [])
    contexts: List[Dict[str, Any]] = []

    for match in matches:
        metadata = match.get("metadata", {}) or {}

        contexts.append({
            "id": match.get("id"),
            "score": float(match.get("score", 0.0)),
            "attack_label": metadata.get("attack_label", "unknown"),
            "dur": metadata.get("dur"),
            "spkts": metadata.get("spkts"),
            "dpkts": metadata.get("dpkts"),
            "metadata": metadata,
        })

    return contexts

def retrieve_context_node(state: RAGState) -> RAGState:
    query = state.get("query", "")
    namespace = state.get("namespace", "network")
    contexts = retrieve_context(query=query, namespace=namespace, k=3)
    return {"retrieved_context": contexts}


def analyze_context_node(state: RAGState) -> RAGState:
    decision = state.get("decision", "UNKNOWN")
    shap_features = state.get("shap_features", [])
    retrieved_context = state.get("retrieved_context", [])

    similar_attacks = []
    reasons = []

    for item in retrieved_context:
        label = item.get("label", 0)
        score = item.get("score", 0.0)

        attack_type = item.get("attack_label", "unknown")

        similar_attacks.append({
            "attack_type": attack_type,
            "similarity": float(score),
            "confidence": "high" if score > 0.8 else "medium" if score > 0.6 else "low"
        })

        reasons.append(
            f"Similar {attack_type} flow: dur={item.get('dur')}, spkts={item.get('spkts')}, similarity={score:.4f}"
        )
    # -------------------------
    # Compute similarity support (Interview Gold)
    # -------------------------
    if similar_attacks:
        avg_similarity = sum(c["similarity"] for c in similar_attacks) / len(similar_attacks)

        if avg_similarity > 0.75:
            support = "strong"
        elif avg_similarity > 0.6:
            support = "moderate"
        else:
            support = "weak"
    else:
        avg_similarity = 0.0
        support = "none"

    rag_analysis = {
    "summary": f"Decision {decision} with {support} supporting evidence.",
    "avg_similarity": avg_similarity,
    "similar_cases": similar_attacks,
    "reasons": reasons,
    "retrieved_matches_count": len(retrieved_context),
    }

    return {"rag_analysis": rag_analysis}

def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("analyze_context", analyze_context_node)

    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "analyze_context")
    graph.add_edge("analyze_context", END)

    return graph.compile()


def get_rag_graph():
    global rag_graph
    if rag_graph is None:
        rag_graph = build_rag_graph()
    return rag_graph


def run_rag_agent(
    query: str,
    decision: str,
    shap_features: List[Dict[str, Any]],
    namespace: str = "network",
) -> Dict[str, Any]:
    graph = get_rag_graph()

    initial_state: RAGState = {
        "query": query,
        "namespace": namespace,
        "decision": decision,
        "shap_features": shap_features,
    }

    result = graph.invoke(initial_state)

    return {
        "rag_context": result.get("retrieved_context", []),
        "rag_analysis": result.get("rag_analysis", {}),
    }


def run_inference(data: dict, with_explanation: bool = False):
    # -------------------------
    # Step 1 — rules
    # -------------------------
    rule_result = evaluate_rules(data)

    matched_rules = rule_result.get("matched_rules", [])
    rule_actions = rule_result.get("rule_actions", [])
    reasons = rule_result.get("reasons", [])
    explanations = rule_result.get("explanations", [])
    attack_types = rule_result.get("attack_hypothesis", [])

    if "BLOCK" in rule_actions:
        return {
            "decision": "BLOCK",
            "decision_source": "RULE",
            "attack_hypothesis": attack_types,
            "matched_rules": matched_rules,
            "reasons": reasons,
            "explanations": explanations,
        }

    if "ALERT" in rule_actions:
        return {
            "decision": "ALERT",
            "decision_source": "RULE",
            "attack_hypothesis": attack_types,
            "matched_rules": matched_rules,
            "reasons": reasons,
            "explanations": explanations,
        }

    # -------------------------
    # Step 2 — model
    # -------------------------
    model, threshold = load_model()

    df = build_features_from_json(data)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    proba = extract_proba(model, df)
    if proba is None:
        return {"error": "Model does not support probability prediction"}

    ml_score = float(proba)

    # -------------------------
    # Decision
    # -------------------------
    if ml_score >= threshold:
        decision = "BLOCK"
    elif ml_score >= threshold * 0.7:
        decision = "ALERT"
    else:
        decision = "ALLOW"

    response: Dict[str, Any] = {
        "prediction": int(ml_score > threshold),
        "ml_score": ml_score,
        "decision": decision,
        "decision_source": "ML",
        "attack_hypothesis": attack_types,
        "matched_rules": matched_rules,
        "reasons": reasons,
    }

    # -------------------------
    # Step 3 — SHAP (optional)
    # -------------------------
    if with_explanation:
        try:
            global explainer

            # Initialize once
            if explainer is None:
                model_for_shap = unwrap_model_for_shap(model)
                explainer = shap.TreeExplainer(model_for_shap)

            shap_vals = explainer.shap_values(df)

            # Binary classification
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            shap_values = shap_vals[0]

            feature_importance = list(zip(df.columns, shap_values, df.iloc[0].values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            top_features = []
            for f, s, v in feature_importance[:5]:
                top_features.append(
                    {
                        "feature": str(f),
                        "value": float(v),
                        "shap_value": float(s),
                        "direction": "increase_risk" if s > 0 else "decrease_risk",
                    }
                )

            response["shap_top_features"] = top_features

        except Exception as e:
            response["explain_error"] = str(e)

    # -------------------------
    # Step 4 — RAG with Pinecone + LangGraph
    # -------------------------
    try:
        shap_features = response.get("shap_top_features", [])
        query = build_query(data, shap_features)

        rag_result = run_rag_agent(
            query=query,
            decision=decision,
            shap_features=shap_features,
            namespace="network-flows",
        )

        rag_analysis = rag_result.get("rag_analysis", {})
        similar_cases = rag_analysis.get("similar_cases", [])
        rag_reasons = rag_analysis.get("reasons", [])

        # attach debug (optional)
        response["rag_query"] = query
        response["rag_context"] = rag_result.get("rag_context", [])
        response["rag_analysis"] = rag_analysis
        response["rag_support"] = rag_analysis.get("summary")
        response["avg_similarity"] = rag_analysis.get("avg_similarity")

        # -------------------------
        # Enrichment (THE IMPORTANT PART)
        # -------------------------
        if not response.get("attack_hypothesis"):
            response["attack_hypothesis"] = list(set(
            case["attack_type"] for case in similar_cases
        ))

        response["reasons"].extend(rag_reasons)
        response["similar_cases"] = similar_cases

    except Exception as e:
        response["rag_error"] = str(e)

    return response