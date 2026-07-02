import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT_DIR.parent

try:
    import anthropic
except ImportError:
    anthropic = None

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

for env_path in (ROOT_DIR / ".env", PROJECT_ROOT / ".env"):
    if env_path.exists():
        load_dotenv(env_path)
        break

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3.2").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
GEMINI_OBSERVER_MODEL = os.getenv("GEMINI_OBSERVER_MODEL", GEMINI_MODEL).strip()
GEMINI_AUDITOR_MODEL = os.getenv("GEMINI_AUDITOR_MODEL", GEMINI_MODEL).strip()
GEMINI_CODER_MODEL = os.getenv("GEMINI_CODER_MODEL", GEMINI_MODEL).strip()

if anthropic and ANTHROPIC_API_KEY:
    CLAUDE_CLIENT = anthropic.Client(api_key=ANTHROPIC_API_KEY)
else:
    CLAUDE_CLIENT = None

if google_genai and GEMINI_API_KEY:
    if GENAI_V2:
        GEMINI_CLIENT = google_genai.Client(api_key=GEMINI_API_KEY)
    else:
        google_genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_CLIENT = google_genai
else:
    GEMINI_CLIENT = None


def _clean_code_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.strip()
    if "```" in cleaned:
        matches = re.findall(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", cleaned, flags=re.DOTALL)
        if matches:
            return matches[0].strip()
        cleaned = cleaned.replace("```python", "").replace("```py", "").replace("```", "")
    return cleaned.strip()


def _clean_json(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0].strip()
    return cleaned


def _provider_order() -> List[str]:
    order = os.getenv("AGENT_PROVIDER_ORDER", os.getenv("AI_PROVIDER_ORDER", "gemini,openrouter,groq"))
    return [provider.strip().lower() for provider in order.split(",") if provider.strip()]


def _role_provider_order(role: str, default: str) -> List[str]:
    env_name = f"{role.upper()}_PROVIDER_ORDER"
    order = os.getenv(env_name, os.getenv("AGENT_PROVIDER_ORDER", os.getenv("AI_PROVIDER_ORDER", default)))
    return [provider.strip().lower() for provider in order.split(",") if provider.strip()]


def _openai_compatible_client(provider: str):
    if provider == "groq" and GROQ_API_KEY:
        return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"), GROQ_MODEL
    if provider == "openrouter" and OPENROUTER_API_KEY:
        headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE", "Universal AI Ecosystem")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
        return (
            OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                default_headers=headers or None,
            ),
            OPENROUTER_MODEL,
        )
    if provider == "openai" and OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), OPENAI_MODEL
    return None, None


def _query_gemini(prompt: str, model: str, max_tokens: int = 900) -> str:
    if not GEMINI_CLIENT:
        raise EnvironmentError("GEMINI_API_KEY is not configured")
    if GENAI_V2:
        response = GEMINI_CLIENT.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.2, "max_output_tokens": max_tokens},
        )
        return getattr(response, "text", "").strip()
    response = GEMINI_CLIENT.generate_text(
        model=model,
        prompt=prompt,
        temperature=0.2,
    )
    return getattr(response, "result", getattr(response, "text", "")).strip()


def _query_openai_compatible(provider: str, prompt: str, max_tokens: int = 900) -> str:
    client, model = _openai_compatible_client(provider)
    if not client:
        raise EnvironmentError(f"{provider} is not configured")
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def _query_provider_chain(prompt: str, providers: List[str], gemini_model: str, max_tokens: int = 900) -> str:
    errors = []
    for provider in providers:
        try:
            if provider in {"gemini", "google", "google_ai_studio"}:
                return _query_gemini(prompt, gemini_model, max_tokens=max_tokens)
            return _query_openai_compatible(provider, prompt, max_tokens=max_tokens)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
    raise EnvironmentError(f"No provider succeeded: {' | '.join(errors)}")


class DomainObserver:
    """
    Phase 1 observer that analyzes a domain using the configured provider chain.
    """

    def __init__(self):
        self.provider_order = _role_provider_order("observer", "gemini,openrouter,groq")

    def _request(self, prompt: str) -> str:
        return _query_provider_chain(prompt, self.provider_order, GEMINI_OBSERVER_MODEL, max_tokens=450)

    def analyze_domain(self, domain: str) -> Dict[str, Any]:
        prompt = f"""
You are a domain observer and analyst.
Analyze the domain '{domain}' deeply.
Return JSON with keys:
  - overview
  - key_concepts
  - learning_sources
  - testing_strategies
  - implementation_approach
  - expected_accuracy
"""
        raw = self._request(prompt)
        cleaned = _clean_json(raw)
        try:
            payload = json.loads(cleaned)
        except Exception:
            payload = {
                "overview": f"Analysis of {domain}.",
                "key_concepts": [domain],
                "learning_sources": ["general documentation"],
                "testing_strategies": ["baseline evaluation"],
                "implementation_approach": "Build a modular agent with shared knowledge integration.",
                "expected_accuracy": 0.7,
            }
        return payload


class ProviderCoderWithSharedKB:
    """
    Phase 2 code generator using the configured provider chain plus shared KB guidance.
    """

    def __init__(self, shared_knowledge_base: Any):
        self.shared_knowledge_base = shared_knowledge_base
        self.client = CLAUDE_CLIENT
        self.provider_order = _role_provider_order("coder", "openrouter,gemini,groq")

    def _ask_claude(self, prompt: str) -> str:
        return _query_provider_chain(prompt, self.provider_order, GEMINI_CODER_MODEL, max_tokens=1200)

    def _ask_direct_anthropic(self, prompt: str) -> str:
        if not self.client:
            raise EnvironmentError("Anthropic Claude client is not configured")
        if hasattr(self.client, "chat"):
            response = self.client.chat.completions.create(
                model=ANTHROPIC_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=900,
            )
            raw = response.choices[0].message["content"]["parts"][0]
            return raw.strip()
        response = self.client.completions.create(
            model=ANTHROPIC_MODEL,
            prompt=prompt,
            max_tokens=900,
            temperature=0.2,
        )
        return response.get("completion", "").strip()

    def _render_module_file(self, code: str) -> str:
        return _clean_code_text(code)

    def generate_knowledge_module(self, domain: str, analysis: Dict[str, Any]) -> str:
        domain_key = domain.lower().replace(" ", "_")
        class_name = f"{domain.title().replace(' ', '')}KnowledgeBase"
        prompt = f"""
Generate a Python knowledge_base.py module for the '{domain}' domain.
Requirements:
- integrate with SharedKnowledgeBase to record every concept with WHAT/WHEN/WHY
- cache learned knowledge locally
- implement learn_from_sources(), answer_query_with_3ws(), get_knowledge(),
  get_concept_knowledge(), learn_from_other_agents(), _ask_what(), _ask_when(), _ask_why()
- use the shared knowledge format and include detailed docstrings.

Domain analysis:
{json.dumps(analysis, indent=2)}
"""
        raw = self._ask_claude(prompt)
        code = self._render_module_file(raw)
        if "class {DOMAIN_CLASS_NAME}KnowledgeBase" in code:
            code = code.replace("{DOMAIN_CLASS_NAME}", class_name)
        return code

    def generate_strategy_tester(self, domain: str, analysis: Dict[str, Any]) -> str:
        prompt = f"""
Create a Python strategy_tester.py for the '{domain}' domain.
It must test strategies using 3-Ws knowledge and return structured results.
Include functions:
- evaluate_strategy()
- backtest_strategy()
- summarize_results()
Use a consistent local cache and support simple rule-based testing.
"""
        raw = self._ask_claude(prompt)
        return self._render_module_file(raw)

    def generate_implementation(self, domain: str, analysis: Dict[str, Any]) -> str:
        class_name = f"{domain.title().replace(' ', '')}Agent"
        prompt = f"""
Generate a main Python agent implementation for the '{domain}' domain.
The file must provide a class named {class_name} with methods:
- __init__()
- bootstrap()
- answer(query)
- learn_from_memory()
It should import local knowledge_base and strategy_tester modules.
"""
        raw = self._ask_claude(prompt)
        return self._render_module_file(raw)

    def generate_intro_file(self, domain: str) -> str:
        class_name = f"{domain.title().replace(' ', '')}Agent"
        code = f"""
from .knowledge_base import {class_name}KnowledgeBase
from .strategy_tester import StrategyTester


class {class_name}:
    def __init__(self, memory_ai=None):
        self.memory_ai = memory_ai
        self.knowledge = {class_name}KnowledgeBase(memory_ai=memory_ai)
        self.tester = StrategyTester()

    def bootstrap(self):
        self.knowledge.learn_from_sources()

    def answer(self, query: str) -> dict:
        return self.knowledge.answer_query_with_3ws(query)

    def learn(self):
        self.bootstrap()
        return self.knowledge.get_knowledge()

    def evaluate(self, strategy: dict):
        return self.tester.evaluate_strategy(strategy, self.knowledge.get_knowledge())
"""
        return code.strip()


class GeminiAuditor:
    """
    Phase 3 auditor that validates agent quality using Gemini.
    """

    def __init__(self):
        self.client = GEMINI_CLIENT
        self.model = GEMINI_AUDITOR_MODEL
        self.provider_order = _role_provider_order("auditor", "gemini,openrouter,groq")

    def _ask_gemini(self, prompt: str) -> str:
        return _query_provider_chain(prompt, self.provider_order, self.model, max_tokens=900)

    def audit_agent(self, domain: str, code_modules: Dict[str, str]) -> Dict[str, Any]:
        prompt = f"""
You are a Gemini auditor.
Review the following generated agent code for domain '{domain}'.
Provide a JSON audit with fields:
- score
- quality
- readability
- security_issues
- improvement_suggestions
"""
        for module_name, code in code_modules.items():
            safe_name = module_name.replace('_', ' ').title()
            prompt += f"\n\n=== {safe_name} ===\n{code[:1200]}\n"

        raw = self._ask_gemini(prompt)
        cleaned = _clean_json(raw)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = {
                "score": 0.0,
                "quality": "Audit could not parse code cleanly.",
                "readability": "unknown",
                "security_issues": [],
                "improvement_suggestions": ["Ensure the generated module is valid Python code."],
            }
        return parsed


class AgentSpawnerWithSharedKB:
    """
    Main spawner class that orchestrates observer, coder, and auditor providers.
    """

    def __init__(self, shared_knowledge_base: Any, agents_dir: str = "agents"):
        self.shared_knowledge_base = shared_knowledge_base
        self.agents_dir = Path(os.path.abspath(agents_dir))
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.observer = DomainObserver()
        self.coder = ProviderCoderWithSharedKB(shared_knowledge_base)
        self.auditor = GeminiAuditor()

    def _validate_domain(self, domain: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_-]", "", domain.lower().replace(" ", "_"))
        if not clean:
            raise ValueError("Invalid domain name")
        return clean

    def _create_agent_folder(self, domain: str) -> Path:
        agent_folder = self.agents_dir / f"{domain}_agent"
        agent_folder.mkdir(parents=True, exist_ok=True)
        init_path = agent_folder / "__init__.py"
        if not init_path.exists():
            init_path.write_text("# Auto-generated agent package\n", encoding="utf-8")
        return agent_folder

    def spawn_agent(self, domain: str) -> Dict[str, Any]:
        domain_key = self._validate_domain(domain)
        if hasattr(self.shared_knowledge_base, "add_agent_source"):
            self.shared_knowledge_base.add_agent_source(domain_key, "universal_ai", f"spawn request for {domain}")
        elif hasattr(self.shared_knowledge_base, "receive_contribution"):
            try:
                self.shared_knowledge_base.receive_contribution(
                    agent_id="universal_ai",
                    domain=domain_key,
                    concept=f"{domain_key}_agent_spawn_request",
                    three_ws={
                        "what": f"Spawn request for {domain_key}",
                        "when": datetime.now().isoformat(),
                        "why": f"User/system interest reached spawn policy for {domain}",
                    },
                    confidence=0.8,
                )
            except Exception:
                pass
        analysis = self.observer.analyze_domain(domain_key)
        knowledge_code = self.coder.generate_knowledge_module(domain_key, analysis)
        strategy_code = self.coder.generate_strategy_tester(domain_key, analysis)
        implementation_code = self.coder.generate_implementation(domain_key, analysis)
        helper_code = self.coder.generate_intro_file(domain_key)

        agent_folder = self._create_agent_folder(domain_key)
        files = {
            "knowledge_base.py": knowledge_code,
            "strategy_tester.py": strategy_code,
            f"{domain_key}_agent.py": implementation_code,
            "agent.py": helper_code,
        }
        for filename, content in files.items():
            self._write_file(agent_folder / filename, content)

        audit = self.auditor.audit_agent(domain_key, files)
        spawn_result = {
            "domain": domain_key,
            "analysis": analysis,
            "files_written": list(files.keys()),
            "audit": audit,
        }
        self._register_spawned_agent(domain_key)
        return spawn_result

    def _write_file(self, path: Path, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _register_spawned_agent(self, domain: str) -> bool:
        try:
            sys.path.insert(0, str(self.agents_dir))
            module = __import__(f"{domain}_agent")
            return hasattr(module, f"{domain.title().replace('_', '').replace('-', '')}Agent")
        except Exception:
            return False

    def list_spawned_agents(self) -> List[str]:
        return [p.name for p in self.agents_dir.iterdir() if p.is_dir() and p.name.endswith("_agent")]


if __name__ == "__main__":
    from core.shared_knowledge_base import SharedKnowledgeBase

    shared = SharedKnowledgeBase()
    spawner = AgentSpawnerWithSharedKB(shared)
    result = spawner.spawn_agent("example")
    print(json.dumps(result, indent=2))
