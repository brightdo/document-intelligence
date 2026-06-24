from groq import Groq
import json
import os
import re
from models import CompanyProfile

def build_prompt(text: str, document_type: str) -> str:
    return f"""You are a business document intelligence system. Extract structured information from the following {document_type} document.

Return ONLY a valid JSON object with these fields (use null for any field you cannot find):
{{
  "company_name": "string or null",
  "sector": "string or null (e.g. fintech, edtech, healthtech)",
  "stage": "string or null (e.g. pre-seed, seed, Series A)",
  "location": "string or null",
  "funding_ask": "string or null (e.g. $2M seed round)",
  "key_metrics": ["list", "of", "strings"] or null,
  "business_summary": "2-3 sentence summary of what the company does",
  "founders": ["list of founder names"] or null,
  "confidence_note": "brief note on data quality or missing information"
}}

Document:
---
{text}
---

Return only the JSON object, no explanation, no markdown fences."""

def extract_profile(text: str, document_type: str = "business") -> CompanyProfile:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=api_key)
    prompt = build_prompt(text, document_type)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a business document intelligence system that extracts structured data and returns only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    data = json.loads(raw)
    return CompanyProfile(**data)