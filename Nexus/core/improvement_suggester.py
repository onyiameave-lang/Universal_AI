import json
import os
from typing import Dict, Optional

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


class ImprovementSuggester:
    """
    Use Gemini to suggest improvements and coordinate with Memory AI.
    """

    def __init__(self, memory_ai=None, market_oracle=None):
        self.memory_ai = memory_ai
        self.market_oracle = market_oracle
        self.client = None
        self.model = os.getenv("GEMINI_IMPROVEMENT_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
        self.provider_order = self._provider_order()
        if google_genai and os.getenv("GEMINI_API_KEY"):
            if GENAI_V2:
                self.client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            else:
                google_genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.client = google_genai

    @staticmethod
    def _provider_order():
        order = os.getenv("IMPROVEMENT_PROVIDER_ORDER", os.getenv("AI_PROVIDER_ORDER", "gemini,openrouter,groq"))
        return [provider.strip().lower() for provider in order.split(",") if provider.strip()]

    @staticmethod
    def _openai_compatible_client(provider: str):
        if provider == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
            headers = {}
            referer = os.getenv("OPENROUTER_HTTP_REFERER")
            title = os.getenv("OPENROUTER_APP_TITLE", "Universal AI Ecosystem")
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-OpenRouter-Title"] = title
            return (
                OpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    default_headers=headers or None,
                ),
                os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
            )
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            return (
                OpenAI(
                    api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1",
                ),
                os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            )
        return None, None

    def _query_provider_chain(self, prompt: str, max_tokens: int) -> str:
        errors = []
        for provider in self.provider_order:
            try:
                if provider in {"gemini", "google", "google_ai_studio"}:
                    if not self.client:
                        raise EnvironmentError("GEMINI_API_KEY is not configured")
                    if GENAI_V2:
                        response = self.client.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config={"temperature": 0.2, "max_output_tokens": max_tokens},
                        )
                        return response.text
                    response = self.client.generate_text(
                        model=self.model,
                        prompt=prompt,
                        temperature=0.2,
                    )
                    return getattr(response, "result", response.text if hasattr(response, "text") else "")

                client, model = self._openai_compatible_client(provider)
                if not client:
                    raise EnvironmentError(f"{provider} is not configured")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
        raise RuntimeError(f"No improvement provider succeeded: {' | '.join(errors)}")

    @staticmethod
    def _clean_response(text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.rfind("```")
            if end > start:
                return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + len("```")
            end = text.rfind("```")
            if end > start:
                return text[start:end].strip()
        return text

    def suggest_improvements(self, domain: str) -> Dict[str, any]:
        market_context = ""
        if self.market_oracle and domain == "trading":
            data = self.market_oracle.get_knowledge()
            market_context = f"\nMarket Oracle Knowledge Base: {json.dumps(list(data.get('concepts', {}).keys()))}"

        prompt = f"""
You are Gemini, a high-quality AI improvement analyst.
Analyze the agent for domain '{domain}' and provide suggestions in JSON.{market_context}
Return:
{{
  "domain": "{domain}",
  "analysis": "...",
  "knowledge_gaps": ["..."],
  "strategy_suggestions": [
    {{
      "strategy": "...",
      "current": "...",
      "improvement": "...",
      "expected_impact": 0.0
    }}
  ],
  "learning_suggestions": ["..."],
  "priority_improvements": ["..."],
  "estimated_improvement": 0.0
}}
"""
        try:
            raw_text = self._query_provider_chain(prompt, max_tokens=800)
            raw_text = self._clean_response(raw_text)
            parsed = json.loads(raw_text)
        except Exception as exc:
            parsed = {
                "domain": domain,
                "analysis": f"Failed to parse Gemini output: {exc}",
                "knowledge_gaps": [],
                "strategy_suggestions": [],
                "learning_suggestions": [],
                "priority_improvements": [],
                "estimated_improvement": 0.0,
            }

        if self.memory_ai and parsed.get("strategy_suggestions"):
            try:
                self.memory_ai.optimize_domain_strategies(domain)
            except Exception:
                pass

        parsed.setdefault("constitutional_alignment", {
            "mission": "optimize and preserve knowledge",
            "evidence": ["memory_retrieval", "strategy_evolution"],
            "confidence": 0.8,
        })
        return parsed

    def analyze_agent_performance(self, domain: str, performance_data: Optional[Dict] = None) -> Dict[str, any]:
        prompt = f"""
Analyze performance data from the '{domain}' agent and suggest optimizations.
Performance data:
{json.dumps(performance_data or {}, indent=2)}

Return JSON with:
{{
  "domain": "{domain}",
  "bottlenecks": ["..."],
  "optimizations": ["..."],
  "learning_rate_improvements": ["..."],
  "resource_improvements": ["..."]
}}
"""
        try:
            raw_text = self._query_provider_chain(prompt, max_tokens=600)
            raw_text = self._clean_response(raw_text)
            return json.loads(raw_text)
        except Exception:
            return {
                "domain": domain,
                "bottlenecks": [],
                "optimizations": [],
                "learning_rate_improvements": [],
                "resource_improvements": [],
            }
