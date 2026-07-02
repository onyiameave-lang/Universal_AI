import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import pdfplumber

try:
    from google import genai as google_genai
    GENAI_V2 = True
except ImportError:
    import google.generativeai as google_genai
    GENAI_V2 = False

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()


class MemoryAIKnowledgeBase:
    """
    Unified Trading Specialist. Learns from YouTube, Books, and internal sources
    using Gemini and integrates with Memory AI.
    """

    def __init__(self, memory_ai=None, cache_dir: str = "knowledge_cache"):
        self.memory_ai = memory_ai
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "raw"), exist_ok=True)
        
        self.local_knowledge_path = os.path.join(self.cache_dir, "system_master_knowledge.json")
        self.local_knowledge = self._load_local_knowledge()
        self.ai_client = self._create_gemini_client()

    def _create_gemini_client(self):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY is required for MarketOracleKnowledgeBase")
        if GENAI_V2:
            return google_genai.Client(api_key=GEMINI_API_KEY)
        else:
            google_genai.configure(api_key=GEMINI_API_KEY)
            return google_genai

    def _ask_gemini(self, prompt: str) -> str:
        try:
            if GENAI_V2:
                response = self.ai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={"temperature": 0.2, "max_output_tokens": 800},
                )
                text = response.text
            else:
                response = self.ai_client.generate_text(
                    model=GEMINI_MODEL,
                    prompt=prompt,
                    temperature=0.2,
                )
                text = getattr(response, "result", response.text if hasattr(response, "text") else "")
            return self._clean_text(text)
        except Exception as exc:
            return f"[Gemini error] {exc}"

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        if "```json" in cleaned:
            start = cleaned.index("```json") + 7
            end = cleaned.rfind("```")
            if end > start:
                return cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.index("```") + 3
            end = cleaned.rfind("```")
            if end > start:
                return cleaned[start:end].strip()
        return cleaned

    def _clean_transcript(self, text: str) -> str:
        if "```" in text:
            text = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), text, flags=re.DOTALL)
        return text.strip()

    def _load_local_knowledge(self) -> Dict[str, Any]:
        if not os.path.exists(self.local_knowledge_path):
            return {"concepts": {}, "sources": []}
        try:
            with open(self.local_knowledge_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"concepts": {}, "sources": []}

    def _save_local_knowledge(self) -> None:
        with open(self.local_knowledge_path, "w", encoding="utf-8") as f:
            json.dump(self.local_knowledge, f, indent=2, ensure_ascii=False)

    def learn_from_youtube(self, video_id: str) -> Dict[str, Any]:
        """Extract knowledge from a YouTube video transcript."""
        raw_path = os.path.join(self.cache_dir, "raw", f"yt_{video_id}.txt")
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                transcript = f.read()
        else:
            try:
                data = YouTubeTranscriptApi.get_transcript(video_id)
                transcript = " ".join([d["text"] for d in data])
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
            except Exception as e:
                return {"error": f"YouTube transcript failed: {e}"}

        prompt = f"Extract key trading concepts from this transcript as a JSON list of strings:\n\n{transcript[:4000]}"
        concepts_raw = self._ask_gemini(prompt)
        try:
            concepts = json.loads(concepts_raw)
            if isinstance(concepts, list):
                results = []
                for c in concepts:
                    results.append(self._learn_concept_internal(c, transcript[:2000], f"YouTube:{video_id}", domain))
                return {"concepts_learned": len(results)}
        except:
            pass
        return {"error": "Failed to parse concepts"}

    def learn_from_pdf(self, pdf_path: str, domain: str = "database") -> Dict[str, Any]:
        """Extract knowledge from a PDF book."""
        if not os.path.exists(pdf_path):
            return {"error": "File not found"}
        
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract from first 5 pages for efficiency in this demonstration
                for page in pdf.pages[:5]:
                    text += page.extract_text() or ""
        except Exception as e:
            return {"error": f"PDF extraction failed: {e}"}

        prompt = f"Extract key trading concepts from this text as a JSON list of strings:\n\n{text[:4000]}"
        concepts_raw = self._ask_gemini(prompt)
        try:
            concepts = json.loads(concepts_raw)
            if isinstance(concepts, list):
                for c in concepts:
                    self._learn_concept_internal(c, text[:2000], os.path.basename(pdf_path))
                return {"concepts_learned": len(concepts)}
        except:
            pass
        return {"error": "Failed to parse concepts"}

    def learn_from_sources(self, youtube_ids: List[str] = None, pdf_paths: List[str] = None) -> None:
        """Batch learning from multiple ecosystem sources."""
        if youtube_ids:
            for vid in youtube_ids:
                self.learn_from_youtube(vid)
        
        if pdf_paths:
            for path in pdf_paths:
                self.learn_from_pdf(path)

        # Default knowledge seeds
        for source in self._default_trading_sources():
            for concept in source["concepts"]:
                self._learn_concept_internal(concept, source["content"], source["title"])

    def _learn_concept_internal(
        self,
        concept: str,
        source_text: str,
        source_title: str,
        domain: str = "trading",
    ) -> Dict[str, Any]:
        if concept.lower() in self.local_knowledge["concepts"]:
            return self.local_knowledge["concepts"][concept.lower()]

        what = self._ask_what(concept, source_text)
        when = self._ask_when(concept, source_text)
        why = self._ask_why(concept, source_text)
        record = {
            "what": what,
            "when": when,
            "why": why,
            "source": source_title,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "confidence": 0.72,
        }
        self.local_knowledge["concepts"][concept.lower()] = record
        self.local_knowledge["sources"].append({"concept": concept, "source": source_title})
        self._save_local_knowledge()

        # Ensure all stored data passes through the AI memory
        if self.memory_ai and hasattr(self.memory_ai, "receive_contribution"):
            try:
                self.memory_ai.receive_contribution(
                    agent_id="knowledge_base",
                    domain=domain,
                    concept=concept,
                    three_ws={"what": what, "when": when, "why": why},
                    confidence=0.72,
                )
            except Exception as e:
                print(f"Memory Sync Error for '{concept}': {e}")

        return record

    def _default_trading_sources(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "RSI Trading Overview",
                "content": "RSI is a momentum indicator used to understand overbought and oversold conditions.",
                "concepts": ["RSI", "momentum", "trend strength"],
            },
            {
                "title": "Moving Averages",
                "content": "Moving averages smooth price data and help identify trend direction and entry points.",
                "concepts": ["moving average", "trend following", "crossover"],
            },
            {
                "title": "Support and Resistance",
                "content": "Support and resistance levels define price areas where buyers or sellers dominate.",
                "concepts": ["support", "resistance", "price levels"],
            },
        ]

    def _ask_what(self, concept: str, source: str) -> str:
        prompt = f"""
Explain WHAT '{concept}' means in trading and how it appears in market analysis.
Use the following source as context:
{source}
"""
        return self._ask_gemini(prompt)

    def _ask_when(self, concept: str, source: str) -> str:
        prompt = f"""
Explain WHEN traders should use '{concept}' and what market conditions make it most valuable.
Use the following source as context:
{source}
"""
        return self._ask_gemini(prompt)

    def _ask_why(self, concept: str, source: str) -> str:
        prompt = f"""
Explain WHY '{concept}' works in trading, including the reasoning and mechanism that makes it effective.
Use the following source as context:
{source}
"""
        return self._ask_gemini(prompt)

    def get_knowledge(self) -> Dict[str, Any]:
        if self.memory_ai:
            try:
                return self.memory_ai.get_domain_knowledge("trading")
            except Exception:
                pass
        return self.local_knowledge

    def get_concept_knowledge(self, concept: str) -> Dict[str, Any]:
        if self.memory_ai:
            try:
                result = self.memory_ai.get_concept("trading", concept)
                if result:
                    return result
            except Exception:
                pass
        return self.local_knowledge["concepts"].get(concept.lower(), {})

    def answer_query_with_3ws(self, query: str) -> Dict[str, Any]:
        concept = self._match_query_to_concept(query)
        if not concept:
            return {
                "query": query,
                "answer": "I did not find a matching trading concept.",
                "what": "",
                "when": "",
                "why": "",
            }
        concept_data = self.get_concept_knowledge(concept)
        return {
            "query": query,
            "concept": concept,
            "what": concept_data.get("what", ""),
            "when": concept_data.get("when", ""),
            "why": concept_data.get("why", ""),
        }

    def _match_query_to_concept(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        for name in self.local_knowledge["concepts"].keys():
            if name in query_lower:
                return name
        for concept in self.local_knowledge["concepts"].keys():
            if any(word in query_lower for word in concept.split()):
                return concept
        return None

    def request_strategy_optimization(self) -> Dict[str, Any]:
        if not self.memory_ai:
            return {"error": "Memory AI not connected"}
        try:
            return self.memory_ai.optimize_domain_strategies("trading")
        except Exception as exc:
            return {"error": str(exc)}


if __name__ == "__main__":
    from pprint import pprint

    oracle = MemoryAIKnowledgeBase()
    oracle.learn_from_sources()
    pprint(oracle.get_knowledge())
