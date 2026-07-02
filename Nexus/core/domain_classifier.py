import json
import os

from openai import OpenAI

try:
    from google import genai as google_genai
    GENAI_V2 = True
except ImportError:
    try:
        import google.generativeai as google_genai
        GENAI_V2 = False
    except ImportError:
        google_genai = None
        GENAI_V2 = False


class DomainClassifier:
    """
    Uses the configured provider chain to intelligently classify query domains.
    Not keyword-based - understands context and nuance.
    """

    def __init__(self):
        self.provider_order = self._load_provider_order()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_CLASSIFIER_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
        self.gemini_client = self._create_gemini_client()
        self.clients = {
            "groq": self._create_client(os.getenv("GROQ_API_KEY"), "https://api.groq.com/openai/v1"),
            "openrouter": self._create_client(os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1"),
            "openai": self._create_client(os.getenv("OPENAI_API_KEY")),
        }
        self.models = {
            "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "openrouter": os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
            "openai": os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        }

    @staticmethod
    def _load_provider_order() -> list:
        order = os.getenv("CLASSIFIER_PROVIDER_ORDER", os.getenv("AI_PROVIDER_ORDER", "gemini,openrouter,groq"))
        return [provider.strip().lower() for provider in order.split(",") if provider.strip()]

    @staticmethod
    def _create_client(api_key: str, base_url: str = None):
        api_key = (api_key or "").strip()
        if not api_key:
            return None
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)

    def _create_gemini_client(self):
        if not self.gemini_api_key or google_genai is None:
            return None
        if GENAI_V2:
            return google_genai.Client(api_key=self.gemini_api_key)
        google_genai.configure(api_key=self.gemini_api_key)
        return google_genai.GenerativeModel(self.gemini_model)

    def _query_gemini(self, prompt: str) -> str:
        if not self.gemini_client:
            raise EnvironmentError("GEMINI_API_KEY is not configured")
        if GENAI_V2:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config={"temperature": 0.15, "max_output_tokens": 220},
            )
        else:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={"temperature": 0.15, "max_output_tokens": 220},
            )
        return getattr(response, "text", "").strip()

    @staticmethod
    def _clean_response(text: str) -> str:
        text = text.strip()
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + len("```")
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        return text

    def classify(self, query: str) -> dict:
        prompt = f"""
You are a domain classifier. Analyze this user query and determine what domain or concept it is about.

Query: "{query}"

Return ONLY valid JSON with these keys:
{{
  "domain": "string (trading, mathematics, chess, security, finance, art, science, general)",
  "confidence": 0.0,
  "reasoning": "Why this domain?",
  "should_spawn_agent": true/false
}}
"""
        text = None
        for provider in self.provider_order:
            if provider in {"gemini", "google", "google_ai_studio"}:
                try:
                    text = self._clean_response(self._query_gemini(prompt))
                    break
                except Exception:
                    continue

            client = self.clients.get(provider)
            if not client:
                continue
            try:
                response = client.chat.completions.create(
                    model=self.models[provider],
                    temperature=0.15,
                    max_tokens=220,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = self._clean_response(response.choices[0].message.content)
                break
            except Exception:
                continue

        if text is None:
            return self._keyword_classify(query)

        try:
            result = json.loads(text)
        except Exception:
            result = self._keyword_classify(query)
        return result

    @staticmethod
    def _keyword_classify(query: str) -> dict:
        query_lower = query.lower()
        domains = {
            "trading": ["trade", "market", "stock", "crypto", "forex"],
            "security": ["security", "vulnerability", "threat", "attack"],
            "finance": ["finance", "money", "portfolio", "investment"],
            "mathematics": ["math", "calculus", "algebra", "equation"],
            "chess": ["chess", "opening", "endgame"],
            "science": ["science", "physics", "chemistry", "biology"],
            "art": ["art", "design", "drawing", "image"],
        }
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    "domain": domain,
                    "confidence": 0.55,
                    "reasoning": "Keyword fallback classifier used because no configured classifier provider succeeded.",
                    "should_spawn_agent": False,
                }
        return {
            "domain": "general",
            "confidence": 0.3,
            "reasoning": "Fallback classifier used because no configured classifier provider succeeded.",
            "should_spawn_agent": False,
        }
