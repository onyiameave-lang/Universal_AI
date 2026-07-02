import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from google import genai as google_genai
    GENAI_V2 = True
except ImportError:
    import google.generativeai as google_genai
    GENAI_V2 = False

class MemoryDatabaseTrainer:
    """
    The 'Brain' of Memory AI. Evaluates data quality, enforces 3-Ws structure,
    and optimizes knowledge for the entire AI ecosystem.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if GENAI_V2:
            self.client = google_genai.Client(api_key=self.api_key)
        else:
            google_genai.configure(api_key=self.api_key)
            self.client = google_genai
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    def evaluate_data_utility(self, domain: str, content: str) -> Dict[str, Any]:
        """Decides if the data is worth storing and how to categorize it."""
        prompt = f"""
        System: You are the Memory AI Database Manager. 
        Evaluate the following data from the '{domain}' agent.
        
        Content: {content}
        
        Return JSON:
        {{
            "should_store": true/false,
            "priority": 0.0-1.0,
            "category": "core_logic/context/noise",
            "reasoning": "why store or discard?"
        }}
        """
        raw = self._query_gemini(prompt)
        try:
            return json.loads(self._clean_json(raw))
        except:
            return {"should_store": True, "priority": 0.5, "category": "context"}

    def transform_to_3ws(self, domain: str, concept: str, context: str) -> Dict[str, str]:
        """Enforces the What/Why/When table structure for all incoming knowledge."""
        prompt = f"""
        Domain: {domain}
        Concept: {concept}
        Context: {context}
        
        Construct a 3-Ws Knowledge Table entry in JSON:
        {{
            "what": "Clear definition of the concept",
            "why": "The underlying reason/mechanism why this is valuable",
            "when": "Specific triggers or conditions when this should be applied"
        }}
        """
        raw = self._query_gemini(prompt)
        try:
            return json.loads(self._clean_json(raw))
        except:
            return {"what": "N/A", "why": "N/A", "when": "N/A"}

    def optimize_shared_knowledge(self, domain: str, current_knowledge: Dict) -> Dict:
        """Periodic optimization of a domain's strategies."""
        prompt = f"""
        Analyze this knowledge base for '{domain}':
        {json.dumps(current_knowledge)}
        
        Identify contradictions, redundant data, and suggest optimizations to improve agent performance.
        Return optimized JSON structure.
        """
        raw = self._query_gemini(prompt)
        return json.loads(self._clean_json(raw))

    def _query_gemini(self, prompt: str) -> str:
        try:
            if GENAI_V2:
                response = self.client.models.generate_content(model=self.model, contents=prompt)
                return response.text
            else:
                response = self.client.generate_text(model=self.model, prompt=prompt)
                return getattr(response, "result", "")
        except Exception as e:
            return f"Error: {e}"

    def _clean_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        return text.strip()

    def print_3w_table(self, knowledge_record: Dict[str, Any]):
        """Creates the visual table showing What, Why, and When."""
        print(f"\n+{'='*80}+")
        print(f"| KNOWLEDGE TABLE: {knowledge_record.get('concept', 'General').upper():<59} |")
        print(f"+{'-'*26}+{'-'*26}+{'-'*26}+")
        print(f"| {'WHAT':<24} | {'WHY':<24} | {'WHEN':<24} |")
        print(f"+{'-'*26}+{'-'*26}+{'-'*26}+")
        
        w = knowledge_record.get('what', 'N/A')[:24]
        y = knowledge_record.get('why', 'N/A')[:24]
        n = knowledge_record.get('when', 'N/A')[:24]
        
        print(f"| {w:<24} | {y:<24} | {n:<24} |")
        print(f"+{'='*80}+\n")