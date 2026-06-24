from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from models import ExtractionRequest, ExtractionResponse
from extractor import extract_profile

load_dotenv()

app = FastAPI(
    title="Document Intelligence API",
    description="Extracts structured company profiles from unstructured business documents using Gemini.",
    version="1.0.0"
)

SAMPLE_DOC = """
TechFlow AI is a seed-stage fintech startup based in San Francisco, CA.
Founded by Sarah Chen and Marcus Williams in 2023, we help small businesses
automate their accounts receivable using AI-powered invoice processing.
We currently process over $50M in invoices monthly across 200 customers,
with 15% MoM growth. We are raising a $3M seed round to expand our
sales team and launch in the UK market. Our gross margin is 72%.
"""

@app.get("/health")
def health():
    return {"status": "ok", "model": "llama-3.3-70b-versatile (Groq)"}

@app.get("/sample")
def sample():
    return {"sample_document": SAMPLE_DOC.strip()}

@app.post("/extract", response_model=ExtractionResponse)
def extract(request: ExtractionRequest):
    try:
        profile = extract_profile(request.text, request.document_type)
        return ExtractionResponse(
            success=True,
            document_type=request.document_type,
            profile=profile,
            raw_text_length=len(request.text)
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        return ExtractionResponse(
            success=False,
            document_type=request.document_type,
            raw_text_length=len(request.text),
            error=str(e)
        )