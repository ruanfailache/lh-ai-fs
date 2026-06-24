from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from document_loader import load_documents
from llm import OpenAIStructuredLLM
from pipeline import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze():
    documents = load_documents()
    if "motion_for_summary_judgment" not in documents:
        raise HTTPException(status_code=500, detail="motion_for_summary_judgment.txt not found")

    report = run_pipeline(documents, llm=OpenAIStructuredLLM())
    return report
