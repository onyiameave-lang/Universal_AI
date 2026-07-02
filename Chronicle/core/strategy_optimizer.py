import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from google import genai as google_genai
    GENAI_V2 = True
except ImportError:
    import google.generativeai as google_genai
    GENAI_V2 = False

from .shared_knowledge_base import SharedKnowledgeBase

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()


class StrategyOptimizer:
    """
    Analyze and optimize strategies learned by agents.
    """

    def __init__(self, shared_knowledge_base: SharedKnowledgeBase, cache_dir: str = "knowledge_cache"):
        self.shared_knowledge_base = shared_knowledge_base
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.ai_client = self._create_gemini_client()
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = {}

    def _create_gemini_client(self):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY is required for StrategyOptimizer")
        if GENAI_V2:
            return google_genai.Client(api_key=GEMINI_API_KEY)
        else:
            google_genai.configure(api_key=GEMINI_API_KEY)
            return google_genai

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        if "```" in text:
            text = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), text, flags=re.DOTALL)
        return text.strip()

    def _query_gemini(self, prompt: str) -> str:
        try:
            if GENAI_V2:
                response = self.ai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={"temperature": 0.2, "max_output_tokens": 900},
                )
                return self._clean_text(response.text)
            else:
                response = self.ai_client.generate_text(
                    model=GEMINI_MODEL,
                    prompt=prompt,
                    temperature=0.2,
                )
                return self._clean_text(getattr(response, "result", response.text if hasattr(response, "text") else ""))
        except Exception as exc:
            raise RuntimeError(f"Gemini query failed: {exc}")

    def _parse_json_response(self, text: str) -> Any:
        cleaned = self._clean_text(text)
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return cleaned

    def optimize_domain_strategies(self, domain: str) -> Dict[str, Any]:
        concepts = self.shared_knowledge_base.get_all_concepts(domain)
        report = {
            "domain": domain,
            "optimized_at": datetime.utcnow().isoformat() + "Z",
            "summary": [],
            "insights": {},
        }

        if not concepts:
            report["summary"].append({
                "message": f"No known concepts found for domain '{domain}'."
            })
            return report

        for concept_name, concept_data in concepts.items():
            optimization = self._optimize_concept(domain, concept_name, concept_data)
            report["summary"].append(optimization)

        insights = self.get_strategy_insights(domain)
        report["insights"] = insights
        self._save_optimization_report(domain, report)
        self.cache_optimized_strategies(domain, report)
        self.optimization_history.setdefault(domain, []).append(report)
        return report

    def _optimize_concept(self, domain: str, concept: str, concept_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
You are a strategy optimizer for the '{domain}' domain.
Analyze the following concept and its current 3-Ws knowledge.

Concept: {concept}
What: {concept_data.get('what')}
When: {concept_data.get('when')}
Why: {concept_data.get('why')}

Answer in JSON with keys:
  - can_be_optimized (true/false)
  - optimization_suggestions (list of strings)
  - optimized_what
  - optimized_when
  - optimized_why
  - effectiveness_improvement (0.0-1.0)
"""
        raw = self._query_gemini(prompt)
        parsed = self._parse_json_response(raw)

        optimization = {
            "concept": concept,
            "original_confidence": float(concept_data.get("confidence", 0.0)),
            "can_be_optimized": False,
            "optimization_suggestions": [],
            "optimized_what": concept_data.get("what"),
            "optimized_when": concept_data.get("when"),
            "optimized_why": concept_data.get("why"),
            "effectiveness_improvement": 0.0,
            "raw_response": raw,
        }

        if isinstance(parsed, dict):
            optimization["can_be_optimized"] = bool(parsed.get("can_be_optimized", False))
            optimization["optimization_suggestions"] = parsed.get("optimization_suggestions", []) or []
            optimization["optimized_what"] = parsed.get("optimized_what") or concept_data.get("what")
            optimization["optimized_when"] = parsed.get("optimized_when") or concept_data.get("when")
            optimization["optimized_why"] = parsed.get("optimized_why") or concept_data.get("why")
            improvement = parsed.get("effectiveness_improvement")
            optimization["effectiveness_improvement"] = float(improvement) if isinstance(improvement, (int, float)) else 0.0

        return optimization

    def test_strategy_effectiveness(self, domain: str, strategy: Dict[str, Any], test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = f"""
You are evaluating a trading strategy for the '{domain}' domain.
Strategy metadata:
{json.dumps(strategy, indent=2)}

Test data summary:
{json.dumps(test_data, indent=2) if test_data else 'None'}

Return JSON:
  - effectiveness_score (0.0-1.0)
  - strengths (list)
  - weaknesses (list)
  - recommended_adjustments (list)
"""
        raw = self._query_gemini(prompt)
        parsed = self._parse_json_response(raw)
        result = {
            "effectiveness_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommended_adjustments": [],
            "raw_response": raw,
        }
        if isinstance(parsed, dict):
            result["effectiveness_score"] = float(parsed.get("effectiveness_score", 0.0))
            result["strengths"] = parsed.get("strengths", []) or []
            result["weaknesses"] = parsed.get("weaknesses", []) or []
            result["recommended_adjustments"] = parsed.get("recommended_adjustments", []) or []
        return result

    def get_strategy_insights(self, domain: str) -> Dict[str, Any]:
        domain_data = self.shared_knowledge_base.get_domain_knowledge(domain)
        strategies = domain_data.get("strategies", {})
        concepts = domain_data.get("concepts", {})

        most_reliable = []
        hidden_gems = []
        needs_improvement = []
        strategy_combinations = []

        for name, details in strategies.items():
            score = float(details.get("effectiveness_score", 0.0))
            usage = int(details.get("usage_count", 0))
            if score >= 0.8 and usage >= 5:
                most_reliable.append(name)
            if score >= 0.75 and usage < 3:
                hidden_gems.append(name)
            if score < 0.6:
                needs_improvement.append(name)

        if len(strategies) >= 2:
            strategy_combinations = self.suggest_strategy_combinations(domain)

        return {
            "most_reliable": most_reliable,
            "hidden_gems": hidden_gems,
            "needs_improvement": needs_improvement,
            "strategy_combinations": strategy_combinations,
            "concept_count": len(concepts),
            "strategy_count": len(strategies),
        }

    def suggest_strategy_combinations(self, domain: str) -> List[Dict[str, Any]]:
        domain_data = self.shared_knowledge_base.get_domain_knowledge(domain)
        strategies = domain_data.get("strategies", {})

        if not strategies:
            return []

        prompt = f"""
As a strategy combination analyst, examine these strategies in the '{domain}' domain:
{json.dumps(list(strategies.keys()), indent=2)}

Return a JSON list of combinations that can work well together with reasons.
Each item should include:
  - strategy_pair
  - reason
  - expected_gain
"""
        raw = self._query_gemini(prompt)
        parsed = self._parse_json_response(raw)
        if isinstance(parsed, list):
            return parsed
        return [{"strategy_pair": list(strategies.keys())[:2], "reason": "Auto-generated combination.", "expected_gain": 0.2}]

    def cache_optimized_strategies(self, domain: str, report: Dict[str, Any]) -> None:
        cache_path = os.path.join(self.cache_dir, f"{domain}_optimized.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _save_optimization_report(self, domain: str, report: Dict[str, Any]) -> None:
        if "optimized_at" not in report:
            report["optimized_at"] = datetime.utcnow().isoformat() + "Z"
        report_path = os.path.join(self.cache_dir, f"{domain}_optimization_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def get_optimization_history(self, domain: str) -> List[Dict[str, Any]]:
        return self.optimization_history.get(domain, [])


class MemoryAIOptimizationAPI:
    """
    API wrapper for other systems to request Memory AI optimization.
    """

    def __init__(self, shared_knowledge_base: SharedKnowledgeBase):
        self.optimizer = StrategyOptimizer(shared_knowledge_base)

    def request_optimization(self, domain: str) -> Dict[str, Any]:
        return self.optimizer.optimize_domain_strategies(domain)

    def request_insights(self, domain: str) -> Dict[str, Any]:
        return self.optimizer.get_strategy_insights(domain)

    def request_test(self, domain: str, strategy: Dict[str, Any], test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.optimizer.test_strategy_effectiveness(domain, strategy, test_data)

    def request_combinations(self, domain: str) -> List[Dict[str, Any]]:
        return self.optimizer.suggest_strategy_combinations(domain)

    def cache_strategies(self, domain: str, report: Dict[str, Any]) -> None:
        self.optimizer.cache_optimized_strategies(domain, report)
