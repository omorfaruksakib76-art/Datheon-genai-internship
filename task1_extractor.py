"""Task 1: LLM Prompt Pipeline & JSON Extractor.

Extracts structured data (invoice or resume) from unstructured text using
Google Gemini, a parameterized few-shot prompt, and strict Pydantic validation.

Usage:
    python task1_extractor.py --type invoice --file sample_invoice.txt
    python task1_extractor.py --type resume  --file sample_resume.txt
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------- Pydantic schemas (strict validation) ----------
class LineItem(BaseModel):
    description: str
    quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)


class Invoice(BaseModel):
    invoice_number: str
    date: Optional[str] = None
    vendor: str
    items: list[LineItem]
    total: float = Field(ge=0)
    currency: str


class Experience(BaseModel):
    company: str
    role: str
    duration: Optional[str] = None


class Resume(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str]
    experience: list[Experience]
    education: list[str]


SCHEMAS = {"invoice": Invoice, "resume": Resume}

# ---------- Few-shot examples ----------
FEW_SHOT = {
    "invoice": (
        "Text: Invoice #A-17 from Blue Pens Ltd dated 2025-01-05. 10 notebooks at 2.5 USD each. Total 25 USD.\n"
        'JSON: {"invoice_number": "A-17", "date": "2025-01-05", "vendor": "Blue Pens Ltd", '
        '"items": [{"description": "notebooks", "quantity": 10, "unit_price": 2.5}], '
        '"total": 25, "currency": "USD"}'
    ),
    "resume": (
        "Text: Rina Das, rina@mail.com. Skills: Python, SQL. Worked at Acme as Data Analyst 2022-2024. BSc CSE, DU.\n"
        'JSON: {"name": "Rina Das", "email": "rina@mail.com", "phone": null, '
        '"skills": ["Python", "SQL"], '
        '"experience": [{"company": "Acme", "role": "Data Analyst", "duration": "2022-2024"}], '
        '"education": ["BSc CSE, DU"]}'
    ),
}

# Parameterized prompt template
PROMPT_TEMPLATE = """You are a precise information-extraction system.
Extract a {doc_type} from the text below and return ONLY valid JSON that matches this JSON schema:
{schema}

Rules:
- Use null for missing optional fields. Do not invent data.
- Numbers must be numbers, not strings.

Example:
{example}

Now extract from this text:
{text}
JSON:"""


def build_prompt(doc_type: str, text: str) -> str:
    return PROMPT_TEMPLATE.format(
        doc_type=doc_type,
        schema=json.dumps(SCHEMAS[doc_type].model_json_schema()),
        example=FEW_SHOT[doc_type],
        text=text,
    )


def extract(doc_type: str, text: str, max_retries: int = 2) -> BaseModel:
    """Call Gemini, validate against the Pydantic schema, retry with error feedback."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY in your .env file.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"response_mime_type": "application/json", "temperature": 0},
    )

    schema = SCHEMAS[doc_type]
    prompt = build_prompt(doc_type, text)
    last_error = None
    for attempt in range(max_retries + 1):
        raw = model.generate_content(prompt).text
        try:
            return schema.model_validate_json(raw)
        except ValidationError as err:
            last_error = err
            print(f"[attempt {attempt + 1}] validation failed, retrying...")
            prompt += f"\n\nYour previous output was invalid:\n{raw}\nErrors:\n{err}\nReturn corrected JSON only."
    raise RuntimeError(f"Could not get valid JSON after retries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured JSON from text with Gemini")
    parser.add_argument("--type", choices=SCHEMAS.keys(), required=True)
    parser.add_argument("--file", required=True, help="Path to a text file")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    result = extract(args.type, text)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
