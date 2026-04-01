from fastapi import APIRouter
from network.schemas import FlowInput
from network.graph import run_inference


router = APIRouter()

@router.get("/")
def root():
    return {"status": "ok"}


@router.post("/predict")
def explain(flow: FlowInput):
    data = flow.model_dump()
    return run_inference(data, with_explanation=True)

