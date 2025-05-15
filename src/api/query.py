from fastapi import APIRouter
from src.retriever import retriever
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class RAGRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/similar_responses", response_model=RAGResponse)
def get_similar_responses(request: RAGRequest):
    results = retriever.get_similar_responses(request.question)
    return RAGResponse(
        answer=results["generated_answer"],
        sources=results["supporting_contexts"]
    )
