import json
import os
import base64
from typing import List, Any

try:
    from google import genai
    NEW_GENAI = True
except ImportError:
    import google.generativeai as genai
    NEW_GENAI = False

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if NEW_GENAI:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    genai.configure(api_key=GEMINI_API_KEY)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def ask_gemini(prompt: str, json_mode: bool = False) -> str:
    """
    Sends a text prompt to Gemini.
    Retries up to 5x with exponential backoff.
    json_mode=True forces pure JSON response — no preamble text.
    """
    if NEW_GENAI:
        cfg = genai.types.GenerateContentConfig(
            temperature=0.2,
            responseMimeType="application/json" if json_mode else None,
        )
        chat = genai_client.chats.create(model=MODEL, config=cfg, history=[])
        response = chat.send_message(prompt)
        return getattr(response, "text", "") or ""

    model = genai.GenerativeModel(MODEL)
    if json_mode:
        cfg = genai.GenerationConfig(temperature=0.2, response_mime_type="application/json")
    else:
        cfg = genai.GenerationConfig(temperature=0.2)
    response = model.generate_content(prompt, generation_config=cfg)
    return response.text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
def ask_gemini_with_images(images_b64: List[str], prompt: str) -> str:
    """
    Sends a batch of base64 JPEG images + prompt to Gemini inline.
    No file upload — avoids ResumableUploadError for large PDFs.
    Used for reading scanned trading books page by page.
    """
    if NEW_GENAI:
        cfg = genai.types.GenerateContentConfig(
            temperature=0.2,
            responseMimeType="application/json",
        )
        chat = genai_client.chats.create(model=MODEL, config=cfg, history=[])
        parts = [genai.types.Part(text=prompt)]
        for img in images_b64:
            parts.append(genai.types.Part(
                inlineData=genai.types.Blob(data=base64.b64decode(img), mimeType="image/jpeg")
            ))
        response = chat.send_message(parts)
        return getattr(response, "text", "") or ""

    model   = genai.GenerativeModel(MODEL)
    content = [prompt]
    for img in images_b64:
        content.append({"mime_type": "image/jpeg", "data": img})
    response = model.generate_content(
        content,
        generation_config=genai.GenerationConfig(temperature=0.2, response_mime_type="application/json")
    )
    return response.text


def parse_json(text: str) -> dict:
    """Safely parses JSON from Gemini — handles markdown fences and preamble."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}