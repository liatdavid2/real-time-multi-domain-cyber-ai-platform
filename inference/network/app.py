from network.schemas import FlowInput
from network.features import build_features_from_json
from network.model import load_model
from network.rules import evaluate_rules
import pandas as pd
import shap
from fastapi import APIRouter

explainer = None



router = APIRouter()

@router.get("/")
def root():
    return {"status": "ok"}


@router.post("/predict")
def explain(flow: FlowInput):
    data = flow.model_dump()
    return run_inference(data, with_explanation=True)


def extract_proba(model, df):
    # direct model
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        return probs[1] if len(probs) > 1 else probs[0]

    # sklearn Pipeline
    if hasattr(model, "steps"):
        last_model = model.steps[-1][1]
        if hasattr(last_model, "predict_proba"):
            probs = last_model.predict_proba(df)[0]
            return probs[1] if len(probs) > 1 else probs[0]

    # fallback
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
            "reasons": reasons,
            "explanations": explanations
        }

    if "ALERT" in rule_actions:
        return {
            "decision": "ALERT",
            "decision_source": "RULE",
            "attack_hypothesis": attack_types,
            "reasons": reasons,
            "explanations": explanations
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

    response = {
        "prediction": int(ml_score > threshold),
        "ml_score": ml_score,
        "decision": decision,
        "decision_source": "ML",
        "attack_hypothesis": attack_types,
        "reasons": reasons
    }

    # -------------------------
    # Step 3 — SHAP (optional)
    # -------------------------
    if with_explanation:
        try:
            global explainer

            # initialize once
            if explainer is None:
                model_for_shap = unwrap_model_for_shap(model)
                explainer = shap.TreeExplainer(model_for_shap)

            shap_vals = explainer.shap_values(df)

            # binary classification
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            shap_values = shap_vals[0]

            feature_importance = list(zip(df.columns, shap_values, df.iloc[0].values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            # -------------------------
            # Build human explanation
            # -------------------------
            top_features = []

            for f, s, v in feature_importance[:5]:
                top_features.append({
                    "feature": str(f),
                    "value": float(v),
                    "shap_value": float(s),
                    "direction": "increase_risk" if s > 0 else "decrease_risk"
                })

            response["shap_top_features"] = top_features

        except Exception as e:
            response["explain_error"] = str(e)
    return response