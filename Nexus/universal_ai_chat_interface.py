# universal_ai_chat_interface.py
# Conversational interface for Universal AI - Talk to the ecosystem naturally

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

try:
    from google import genai as google_genai
    GEMINI_V2 = True
except ImportError:
    try:
        import google.generativeai as google_genai
        GEMINI_V2 = False
    except ImportError:
        google_genai = None
        GEMINI_V2 = False

# Attempt to load .env from multiple locations
env_paths = [
    os.path.join(os.path.dirname(__file__), '.env'),
    os.path.join(os.path.dirname(__file__), '..', '.env')
]
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break


class UserIntent(Enum):
    """Types of user intents."""
    SPAWN_AGENT = "spawn_agent"
    QUERY_KNOWLEDGE = "query_knowledge"
    CHECK_STATUS = "check_status"
    MODIFY_SETTINGS = "modify_settings"
    ANALYZE_DATA = "analyze_data"
    GET_INSIGHTS = "get_insights"


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    intent: Optional[UserIntent] = None
    agent_spawned: Optional[str] = None


class UniversalAIChatInterface:
    """
    Natural language interface for Universal AI ecosystem.
    Allows users to chat with the system to spawn agents, query knowledge, etc.
    """

    def __init__(self, memory_ai_system=None, universal_ai_system=None, api_key: str = None):
        """
        Initialize the chat interface.
        
        Args:
            memory_ai_system: Reference to Memory AI system
            universal_ai_system: Reference to Universal AI system
            api_key: Optional Anthropic API key for legacy/provider-order use
        """
        self.memory_ai = memory_ai_system
        self.universal_ai = universal_ai_system
        
        # Ensure API keys are valid and stripped of whitespace/quotes.
        self.anthropic_api_key = self._clean_api_key(api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.openai_api_key = self._clean_api_key(os.getenv("OPENAI_API_KEY"))
        self.groq_api_key = self._clean_api_key(os.getenv("GROQ_API_KEY"))
        self.gemini_api_key = self._clean_api_key(os.getenv("GEMINI_API_KEY"))
        self.openrouter_api_key = self._clean_api_key(os.getenv("OPENROUTER_API_KEY"))
        self.provider_order = self._load_provider_order()
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.openai_model = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.gemini_model = os.getenv("GEMINI_CHAT_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

        self.client = anthropic.Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        self.groq_client = self._create_openai_compatible_client(
            self.groq_api_key,
            "https://api.groq.com/openai/v1",
        )
        self.gemini_client = self._create_gemini_client()
        self.openrouter_client = self._create_openrouter_client()
        
        self.conversation_history: List[Message] = []
        self.system_prompt = self._build_system_prompt()
        self.mission_context = {
            "mode": "conversation",
            "purpose": "assist humanity through coordinated agents",
            "evidence": [],
        }
        self.last_spawn_result = None
        self.last_spawn_error = None
        self.last_agent_domain = None

    @staticmethod
    def _clean_api_key(api_key: Optional[str]) -> Optional[str]:
        if not api_key:
            return None
        return api_key.strip().replace('"', '').replace("'", "") or None

    @staticmethod
    def _load_provider_order() -> List[str]:
        order = os.getenv("AI_PROVIDER_ORDER", "gemini,openrouter,groq")
        providers = [provider.strip().lower() for provider in order.split(",") if provider.strip()]
        return providers or ["gemini", "openrouter", "groq"]
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the active chat provider."""
        return """You are an intelligent AI assistant managing a Universal AI Ecosystem.

Your capabilities:
1. SPAWN AGENTS: When a user describes a new domain/task, you can spawn specialized AI agents
2. QUERY KNOWLEDGE: Access the Memory AI knowledge base to answer questions
3. PROVIDE INSIGHTS: Analyze system performance and offer recommendations
4. MANAGE AGENTS: View agent status, performance, and manage the agent network

Important existing agents:
- Trading, forex, market, and symbol-analysis requests are handled by the existing MarketOracle trading system.
- Do not claim that you spawned a trading agent. Say you are routing the request to MarketOracle.
- Only say an agent was spawned when the backend confirms it.

When a user asks you something, analyze if you need to:
- SPAWN a new agent (for new domains/capabilities)
- USE existing agents (query or delegate tasks)
- ACCESS knowledge base (answer from existing knowledge)
- ANALYZE current state (provide system insights)

Always respond conversationally and naturally. Be helpful and proactive.

When the user describes a new capability they need, analyze:
1. What domain is this? (trading, security, gaming, NLP, etc.)
2. Do we have an agent for this? If yes, use it.
3. What knowledge would this agent need? Reference Memory AI.
4. What would be the agent's role/purpose?

Format responses naturally - don't use bullet points unless necessary.
Do not decide by yourself that an agent was spawned. The backend spawns agents only when the user explicitly asks for an agent or when repeated interest reaches the configured threshold.

Example flow:
User: "I need something that analyzes Twitter sentiment about crypto"
You: "That sounds like a sentiment-analysis domain. I can use existing knowledge first; if you want a dedicated agent, ask me to spawn one, or I can track repeated demand and create one when it crosses the threshold."
"""

    def process_user_message(self, user_input: str) -> Tuple[str, Optional[UserIntent], Optional[str]]:
        """
        Process a user message and return AI response.
        
        Args:
            user_input: The user's message
            
        Returns:
            Tuple of (response_text, detected_intent, spawned_agent_name)
        """
        
        # Add user message to history
        user_message = Message(
            role="user",
            content=user_input,
            timestamp=datetime.now().isoformat()
        )
        self.conversation_history.append(user_message)
        self.mission_context["evidence"].append(user_input)

        trading_response = self._try_handle_trading_request(user_input)
        if trading_response:
            assistant_message = Message(
                role="assistant",
                content=trading_response,
                timestamp=datetime.now().isoformat(),
                intent=UserIntent.ANALYZE_DATA,
                agent_spawned=None
            )
            self.conversation_history.append(assistant_message)
            return trading_response, UserIntent.ANALYZE_DATA, None

        status_response = self._try_handle_agent_status_request(user_input)
        if status_response:
            assistant_message = Message(
                role="assistant",
                content=status_response,
                timestamp=datetime.now().isoformat(),
                intent=UserIntent.CHECK_STATUS,
                agent_spawned=None
            )
            self.conversation_history.append(assistant_message)
            return status_response, UserIntent.CHECK_STATUS, None

        explicit_spawn = self._try_handle_explicit_spawn_request(user_input)
        if explicit_spawn:
            response_text, agent_name = explicit_spawn
            assistant_message = Message(
                role="assistant",
                content=response_text,
                timestamp=datetime.now().isoformat(),
                intent=UserIntent.SPAWN_AGENT,
                agent_spawned=agent_name
            )
            self.conversation_history.append(assistant_message)
            return response_text, UserIntent.SPAWN_AGENT, agent_name
        
        # Build messages for the active provider.
        messages = [
            {"role": m.role, "content": m.content}
            for m in self.conversation_history
        ]
        
        # Get response from the configured model provider.
        try:
            assistant_response = self._get_ai_response(messages)
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error communicating with my brain: {e}"
            return error_msg, None, None
        
        # Analyze the response for intents and actions.
        intent = self._analyze_intent(user_input, assistant_response)
        model_requested_spawn = self._check_for_agent_spawn(assistant_response)
        spawn_decision = self._evaluate_spawn_policy(user_input, assistant_response)
        
        # Spawn only when policy allows it, not just because the model said so.
        agent_name = None
        if spawn_decision and self.universal_ai:
            agent_name = self._spawn_agent(spawn_decision)
            assistant_response = self._inject_spawn_confirmation(
                assistant_response, 
                spawn_decision, 
                agent_name
            )
            if agent_name:
                intent = UserIntent.SPAWN_AGENT
        elif model_requested_spawn:
            assistant_response = self._remove_spawn_markers(assistant_response)
            assistant_response += (
                "\n\nI have not spawned a new agent yet. I only spawn agents when you explicitly ask for one "
                "or when the same missing domain reaches the repeated-interest threshold."
            )
        
        # Add assistant message to history
        assistant_message = Message(
            role="assistant",
            content=assistant_response,
            timestamp=datetime.now().isoformat(),
            intent=intent,
            agent_spawned=agent_name
        )
        self.conversation_history.append(assistant_message)

        # Store conversation and agent work in Memory AI for long-term learning
        if self.memory_ai:
            try:
                if hasattr(self.memory_ai, "receive_contribution"):
                    self.memory_ai.receive_contribution(
                        agent_id="chat_interface",
                        domain="conversation",
                        concept=f"chat_{intent.value if intent else 'query'}_{datetime.now().timestamp()}",
                        three_ws={
                            "what": f"User asked: {user_input}",
                            "when": "During Universal AI chat conversation",
                            "why": f"Assistant answered: {assistant_response}",
                        },
                        confidence=0.9
                    )
            except Exception as e:
                print(f"âœ— Memory AI persistence error: {e}")
        
        self.mission_context.update({
            "last_intent": intent.value if intent else None,
            "last_agent": agent_name,
            "mode": "conversation",
            "confidence": 0.85 if assistant_response else 0.0,
        })
        return assistant_response, intent, agent_name

    def _try_handle_explicit_spawn_request(self, user_input: str) -> Optional[Tuple[str, Optional[str]]]:
        if not self._is_explicit_agent_request(user_input):
            return None
        if not self.universal_ai:
            return "The Universal AI backend is not connected, so I cannot spawn an agent yet.", None

        spawn_decision = self._evaluate_spawn_policy(user_input, "")
        if not spawn_decision:
            domain = self._extract_requested_domain(user_input) or "unknown"
            if domain != "unknown":
                self.last_agent_domain = domain
            return (
                f"I did not spawn a new agent for `{domain}` because an existing agent already covers that domain "
                "or the request did not resolve to a concrete domain.",
                None,
            )

        agent_name = self._spawn_agent(spawn_decision)
        domain = spawn_decision.get("domain", "unknown")
        if domain != "unknown":
            self.last_agent_domain = domain
        agent_folder = os.path.join(self.universal_ai.agents_dir, f"{domain}_agent")

        if agent_name:
            files = []
            if isinstance(self.last_spawn_result, dict):
                files = self.last_spawn_result.get("files_written", [])
            return (
                f"Backend spawn complete for `{domain}`.\n\n"
                f"Agent: {agent_name}\n"
                f"Folder: {agent_folder}\n"
                f"Files: {', '.join(files) if files else 'created'}\n"
                "I also attempted to bootstrap/train the agent against Memory AI.",
                agent_name,
            )

        reason = self.last_spawn_error or "The backend did not create the expected files."
        return (
            f"I tried to spawn a `{domain}` agent, but it is not ready because the backend failed before creating a verified agent folder.\n\n"
            f"Expected folder: {agent_folder}\n"
            f"Reason: {reason}",
            None,
        )

    def _try_handle_agent_status_request(self, user_input: str) -> Optional[str]:
        text = user_input.lower()
        if "agent" not in text or not any(word in text for word in ["ready", "status", "created", "spawned", "done"]):
            return None
        if not self.universal_ai:
            return "The Universal AI backend is not connected, so I cannot check agent status."

        domain = self._extract_requested_domain(user_input) or self.last_agent_domain
        if not domain:
            agents = []
            if getattr(self.universal_ai, "agent_registry", None):
                agents = sorted(self.universal_ai.agent_registry.list_agents().keys())
            spawned_dirs = self._list_spawned_agent_dirs()
            return (
                "I checked the backend agent registry and filesystem.\n\n"
                f"Registered domains: {', '.join(agents) if agents else 'none'}\n"
                f"Spawned folders: {', '.join(spawned_dirs) if spawned_dirs else 'none'}"
            )

        agent_folder = os.path.join(self.universal_ai.agents_dir, f"{domain}_agent")
        exists = os.path.isdir(agent_folder)
        files = sorted(os.listdir(agent_folder)) if exists else []
        registered = (
            getattr(self.universal_ai, "agent_registry", None)
            and self.universal_ai.agent_registry.has_agent(domain)
        )

        if exists:
            return (
                f"The `{domain}` agent folder exists.\n\n"
                f"Folder: {agent_folder}\n"
                f"Files: {', '.join(files) if files else 'none'}\n"
                f"Registered: {'yes' if registered else 'not yet'}"
            )

        reason = ""
        if domain == self.last_agent_domain and self.last_spawn_error:
            reason = f"\nLast backend error: {self.last_spawn_error}"

        return (
            f"No verified `{domain}` agent exists in this backend yet.\n\n"
            f"Expected folder: {agent_folder}\n"
            "So if the chat said it was 'almost ready', that was not backed by a real folder or registered agent."
            f"{reason}"
        )

    def _evaluate_spawn_policy(self, user_input: str, assistant_response: str) -> Optional[Dict]:
        if not self.universal_ai:
            return None

        explicit = self._is_explicit_agent_request(user_input)
        model_request = self._check_for_agent_spawn(assistant_response)

        try:
            classification = self.universal_ai.classifier.classify(user_input)
        except Exception:
            classification = {"domain": "general", "confidence": 0.0}

        domain = (self._extract_requested_domain(user_input) or classification.get("domain") or "general").lower()
        if domain in {"", "general", "conversation"}:
            return None

        self.last_agent_domain = domain

        if getattr(self.universal_ai, "agent_registry", None) and self.universal_ai.agent_registry.has_agent(domain):
            return None

        current_count = self.universal_ai.query_counts.get(domain, 0) + 1
        self.universal_ai.query_counts[domain] = current_count
        threshold = int(os.getenv("AGENT_SPAWN_THRESHOLD", "5"))

        if not explicit and current_count < threshold:
            return None

        requested_name = model_request.get("name") if model_request else None
        agent_name = requested_name or f"{domain.title().replace('_', '')}Agent"

        return {
            "name": agent_name,
            "domain": domain,
            "type": "explicit" if explicit else "threshold",
            "query_count": current_count,
            "classification": classification,
        }

    @staticmethod
    def _extract_requested_domain(user_input: str) -> Optional[str]:
        text = user_input.lower()
        domain_keywords = {
            "chess": ("chess", "opening", "endgame"),
            "trading": ("trading", "forex", "eurusd", "marketoracle"),
            "security": ("security", "vulnerability", "threat"),
            "nlp": ("nlp", "language", "sentiment"),
            "gaming": ("game", "gaming"),
            "data": ("data", "analytics", "visualization"),
            "research": ("research", "study"),
        }
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain
        return None

    def _list_spawned_agent_dirs(self) -> List[str]:
        if not self.universal_ai or not getattr(self.universal_ai, "agents_dir", None):
            return []
        if not os.path.isdir(self.universal_ai.agents_dir):
            return []
        return sorted(
            name for name in os.listdir(self.universal_ai.agents_dir)
            if name.endswith("_agent") and os.path.isdir(os.path.join(self.universal_ai.agents_dir, name))
        )

    @staticmethod
    def _is_explicit_agent_request(user_input: str) -> bool:
        text = user_input.lower()
        phrases = (
            "spawn an agent",
            "spawn agent",
            "create an agent",
            "create agent",
            "build an agent",
            "build agent",
            "make an agent",
            "make agent",
            "new agent",
        )
        if any(phrase in text for phrase in phrases):
            return True
        if "agent" not in text:
            return False
        request_words = (
            "need", "want", "would like", "can i get", "please add",
            "set up", "setup", "generate", "develop", "create", "build",
            "make", "spawn"
        )
        return any(word in text for word in request_words)

    def _try_handle_trading_request(self, user_input: str) -> Optional[str]:
        if not self.universal_ai:
            return None

        if not self._is_trading_request(user_input):
            return None

        try:
            response = None
            if getattr(self.universal_ai, "agent_registry", None):
                trading_agent = self.universal_ai.agent_registry.get_agent("trading")
                if trading_agent and hasattr(trading_agent, "answer"):
                    response = trading_agent.answer(user_input)

            if response is None:
                result = self.universal_ai.process_query(user_input)
                if result.get("domain") != "trading":
                    return None
                response = result.get("response", {})

            if isinstance(response, dict):
                return self._format_trading_response(response)
            return str(response)
        except Exception as exc:
            return (
                "I tried to route that to your existing MarketOracle trading bot, "
                f"but the analysis call failed: {exc}"
            )

    @staticmethod
    def _is_trading_request(user_input: str) -> bool:
        text = user_input.lower()
        trading_terms = (
            "eurusd", "eur/usd", "gbpusd", "gbp/usd", "btcusd", "btc/usd",
            "ethusd", "eth/usd", "forex", "trade", "trading", "market",
            "price action", "analyze symbol", "marketoracle"
        )
        return any(term in text for term in trading_terms)

    @staticmethod
    def _format_trading_response(response: Dict) -> str:
        if response.get("agent") != "MarketOracle":
            return json.dumps(response, indent=2)

        if response.get("status") == "not_found":
            available = ", ".join(response.get("available_symbols", [])[:12])
            return (
                f"I routed this to your existing MarketOracle trading bot, but it does not have cached data "
                f"for {response.get('requested_symbol')} ({response.get('normalized_symbol')}). "
                f"Available symbols include: {available}."
            )

        if response.get("status") == "available":
            available = ", ".join(response.get("available_symbols", [])[:12])
            return (
                "Your existing MarketOracle trading bot is registered and available in analysis-only mode. "
                f"Available cached symbols include: {available}."
            )

        characteristics = ", ".join(response.get("overall_characteristics", [])) or "not enough characteristics detected"
        trend = response.get("trend_strength", {})
        volatility = response.get("volatility_profile", {})
        trade_idea = response.get("trade_idea", {})
        symbol = response.get("symbol")
        strategy_name = response.get("strategy_name") or "cached/default strategy"
        action = trade_idea.get("action", "NO_TRADE")
        trade_lines = [
            f"Trade idea: {action}",
            f"Confidence: {trade_idea.get('confidence', 0.0)}",
            f"Timeframe: {trade_idea.get('timeframe')}",
            f"Entry reference: {trade_idea.get('entry_reference')}",
            f"Stop loss: {trade_idea.get('stop_loss')}",
            f"Take profit: {trade_idea.get('take_profit')}",
            f"Reason: {trade_idea.get('reason')}",
        ]

        return (
            f"I routed this to your existing MarketOracle trading bot instead of spawning a new agent.\n\n"
            f"Symbol: {symbol}\n"
            f"Symbol type: {response.get('symbol_type')}\n"
            f"Overall characteristics: {characteristics}\n"
            f"Strategy loaded: {'yes' if response.get('strategy_loaded') else 'no'} ({strategy_name})\n\n"
            + "\n".join(trade_lines) +
            "\n\n"
            f"Trend strength by timeframe:\n{json.dumps(trend, indent=2)}\n\n"
            f"Volatility profile:\n{json.dumps(volatility, indent=2)}\n\n"
            "Mode: analysis-only. I did not start live trading or place any MT5 order."
        )

    def _get_ai_response(self, messages: List[Dict[str, str]]) -> str:
        """Try configured providers in order, then use a local fallback."""
        errors = []
        provider_calls = {
            "groq": self._call_groq,
            "gemini": self._call_gemini,
            "google": self._call_gemini,
            "google_ai_studio": self._call_gemini,
            "openrouter": self._call_openrouter,
            "anthropic": self._call_anthropic,
            "claude": self._call_anthropic,
            "openai": self._call_openai,
            "chatgpt": self._call_openai,
        }

        for provider in self.provider_order:
            call_provider = provider_calls.get(provider)
            if not call_provider:
                errors.append(f"{provider}: unknown provider")
                continue

            try:
                return call_provider(messages)
            except EnvironmentError as exc:
                errors.append(f"{provider}: {exc}")
            except Exception as exc:
                errors.append(f"{provider}: {exc}")

        fallback = self._local_fallback_response(messages[-1]["content"])
        if errors:
            return f"{fallback}\n\nProvider fallback details: {' | '.join(errors)}"
        return fallback

    def _call_anthropic(self, messages: List[Dict[str, str]]) -> str:
        if not self.client:
            raise EnvironmentError("ANTHROPIC_API_KEY is not configured")
        response = self.client.messages.create(
            model=self.anthropic_model,
            max_tokens=1000,
            system=self.system_prompt,
            messages=messages
        )
        return response.content[0].text

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        if not self.openai_client:
            raise EnvironmentError("OPENAI_API_KEY is not configured")
        return self._call_openai_compatible(self.openai_client, self.openai_model, messages)

    def _call_groq(self, messages: List[Dict[str, str]]) -> str:
        if not self.groq_client:
            raise EnvironmentError("GROQ_API_KEY is not configured")
        return self._call_openai_compatible(self.groq_client, self.groq_model, messages)

    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        if not self.openrouter_client:
            raise EnvironmentError("OPENROUTER_API_KEY is not configured")
        return self._call_openai_compatible(self.openrouter_client, self.openrouter_model, messages)

    def _call_openai_compatible(self, client, model: str, messages: List[Dict[str, str]]) -> str:
        provider_messages = [{"role": "system", "content": self.system_prompt}] + messages
        response = client.chat.completions.create(
            model=model,
            max_tokens=1000,
            messages=provider_messages,
        )
        return response.choices[0].message.content

    @staticmethod
    def _create_openai_compatible_client(api_key: Optional[str], base_url: str, default_headers=None):
        if not api_key:
            return None
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    def _create_openrouter_client(self):
        headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE", "Universal AI Ecosystem")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
        return self._create_openai_compatible_client(
            self.openrouter_api_key,
            "https://openrouter.ai/api/v1",
            headers or None,
        )

    def _create_gemini_client(self):
        if not self.gemini_api_key or google_genai is None:
            return None
        if GEMINI_V2:
            return google_genai.Client(api_key=self.gemini_api_key)
        google_genai.configure(api_key=self.gemini_api_key)
        return google_genai.GenerativeModel(self.gemini_model)

    def _call_gemini(self, messages: List[Dict[str, str]]) -> str:
        if not self.gemini_client:
            raise EnvironmentError("GEMINI_API_KEY is not configured")
        prompt = self._format_gemini_prompt(messages)
        if GEMINI_V2:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config={"temperature": 0.2, "max_output_tokens": 1000},
            )
        else:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 1000},
            )
        return getattr(response, "text", "").strip()

    def _format_gemini_prompt(self, messages: List[Dict[str, str]]) -> str:
        conversation = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in messages
        )
        return f"{self.system_prompt}\n\nConversation:\n{conversation}\n\nAssistant:"

    @staticmethod
    def _should_fallback_to_openai(error: Exception) -> bool:
        error_text = str(error).lower()
        fallback_markers = (
            "credit balance",
            "billing",
            "quota",
            "insufficient",
            "rate limit",
        )
        return any(marker in error_text for marker in fallback_markers)

    @classmethod
    def _should_use_local_fallback(cls, error: Exception) -> bool:
        error_text = str(error).lower()
        fallback_markers = (
            "credit balance",
            "billing",
            "quota",
            "insufficient_quota",
            "insufficient",
            "rate limit",
        )
        return any(marker in error_text for marker in fallback_markers)

    def _local_fallback_response(self, user_input: str) -> str:
        """Provide a useful no-network response when model providers are unavailable."""
        user_lower = user_input.lower()

        if any(word in user_lower for word in ["status", "health", "ready", "running"]):
            memory_status = "connected" if self.memory_ai else "unavailable"
            universal_status = "connected" if self.universal_ai else "unavailable"
            return (
                "The local Universal AI API is running. "
                f"Memory AI is {memory_status}, and the Universal AI orchestrator is {universal_status}. "
                "External model calls are unavailable because the configured provider quota or billing is exhausted."
            )

        if any(word in user_lower for word in ["spawn", "create", "build", "new agent", "need an agent"]):
            return (
                "I can recognize this as an agent-spawn request, but external model quota is exhausted, "
                "so I cannot safely generate or audit a new specialized agent right now. "
                "You can still use the direct spawn endpoint after billing/quota is restored."
            )

        if any(word in user_lower for word in ["knowledge", "memory", "concept", "database"]):
            return (
                "The local Memory AI service is available, but natural-language synthesis is currently limited "
                "because both configured model providers are out of quota. "
                "Use /api/knowledge or /api/knowledge/query for direct knowledge-base access."
            )

        return (
            "The Universal AI API received your message, but the configured AI providers are currently "
            "unavailable or not configured. Local services are still running; add a working Groq, Gemini, "
            "or OpenRouter key to enable full chat responses."
        )

    def _analyze_intent(self, user_input: str, assistant_response: str) -> UserIntent:
        """Analyze user intent from the input and response."""
        
        user_lower = user_input.lower()
        response_lower = assistant_response.lower()
        
        # Check for spawn intent
        if any(word in user_lower for word in ["spawn", "create", "new", "need", "build"]):
            if any(word in response_lower for word in ["spawn", "creating", "agent"]):
                return UserIntent.SPAWN_AGENT
        
        # Check for knowledge query
        if any(word in user_lower for word in ["what", "how", "explain", "tell", "know", "learn"]):
            return UserIntent.QUERY_KNOWLEDGE
        
        # Check for status
        if any(word in user_lower for word in ["status", "how are", "check", "show"]):
            return UserIntent.CHECK_STATUS
        
        # Check for analysis
        if any(word in user_lower for word in ["analyze", "analyze", "performance", "metrics"]):
            return UserIntent.ANALYZE_DATA
        
        # Check for insights
        if any(word in user_lower for word in ["insight", "suggest", "recommend", "improve"]):
            return UserIntent.GET_INSIGHTS
        
        return UserIntent.QUERY_KNOWLEDGE

    def _check_for_agent_spawn(self, response: str) -> Optional[Dict]:
        """
        Check if the response indicates an agent should be spawned.
        Look for patterns like [SPAWN_AGENT: AgentName] in the response.
        """
        
        # Look for explicit spawn markers
        if "[SPAWN_AGENT:" in response:
            start = response.find("[SPAWN_AGENT:") + len("[SPAWN_AGENT:")
            end = response.find("]", start)
            agent_name = response[start:end].strip()
            return {"name": agent_name, "type": "explicit"}
        
        return None

    def _spawn_agent(self, spawn_info: Dict) -> str:
        """
        Spawn a new agent based on the spawn info.
        
        Args:
            spawn_info: Dictionary with agent spawn details
            
        Returns:
            Name of the spawned agent
        """
        
        if not self.universal_ai:
            print("âš ï¸ Universal AI system not connected")
            return None
        
        agent_name = spawn_info.get("name", "UnknownAgent")
        self.last_spawn_result = None
        self.last_spawn_error = None
        
        try:
            domain = (spawn_info.get("domain") or "general").lower()

            if (
                getattr(self.universal_ai, "agent_registry", None)
                and self.universal_ai.agent_registry.has_agent(domain)
            ):
                print(f"Using existing agent for domain: {domain}")
                return None
            
            # Spawn agent
            # Note: The UniversalAI class in main.py uses agent_spawner.spawn_agent(domain)
            # We adapt to the existing architecture:
            self.universal_ai.query_counts[domain] = 5 # Force threshold
            result = self.universal_ai.agent_spawner.spawn_agent(domain)
            self.last_spawn_result = result

            agent_folder = os.path.join(
                self.universal_ai.agents_dir,
                f"{domain}_agent"
            )
            files_written = result.get("files_written", []) if isinstance(result, dict) else []
            if not os.path.isdir(agent_folder) or not files_written:
                self.last_spawn_error = "Agent spawner returned without creating verified files."
                print(f"Agent spawn did not create files for domain: {domain}")
                return None

            if getattr(self.universal_ai, "agent_registry", None):
                self.universal_ai.agent_registry.register_spawned_agent(domain)
            self._train_spawned_agent(domain)
            self.universal_ai.query_counts[domain] = 0
            
            print(f"âœ“ Agent spawned: {agent_name} (Domain: {domain})")
            return agent_name
            
        except Exception as e:
            self.last_spawn_error = str(e)
            print(f"âœ— Error spawning agent: {e}")
            return None

    def _train_spawned_agent(self, domain: str) -> None:
        if not getattr(self.universal_ai, "agent_registry", None):
            return
        agent = self.universal_ai.agent_registry.get_agent(domain)
        if not agent:
            return
        for method_name in ("bootstrap", "learn", "learn_from_memory"):
            method = getattr(agent, method_name, None)
            if callable(method):
                try:
                    method()
                except TypeError:
                    try:
                        method(self.memory_ai)
                    except Exception:
                        pass
                except Exception:
                    pass
                return

    def _inject_spawn_confirmation(
        self, 
        response: str, 
        spawn_info: Dict, 
        agent_name: Optional[str]
    ) -> str:
        """
        Inject confirmation of agent spawn into the response.
        """
        
        marker = f"[SPAWN_AGENT: {spawn_info.get('name')}]"
        response = response.replace(marker, "")

        if agent_name:
            confirmation = f"\n\n**{agent_name} spawned successfully.**"
            response += confirmation
        else:
            response += (
                "\n\nI did not create a new agent. If this was a trading request, "
                "I used the existing MarketOracle trading system instead."
            )
        
        return response

    @staticmethod
    def _remove_spawn_markers(response: str) -> str:
        while "[SPAWN_AGENT:" in response:
            start = response.find("[SPAWN_AGENT:")
            end = response.find("]", start)
            if end == -1:
                break
            response = response[:start] + response[end + 1:]
        return response.strip()

    def multi_turn_conversation(self) -> None:
        """
        Start an interactive multi-turn conversation.
        Type 'exit' to quit, 'history' to see conversation, 'clear' to reset.
        """
        
        print("\n" + "="*80)
        print("ðŸ¤– UNIVERSAL AI ECOSYSTEM - CHAT INTERFACE")
        print("="*80)
        print("\nYou can now chat naturally with your AI ecosystem!")
        print("- Ask it to spawn agents for new tasks")
        print("- Query the knowledge base")
        print("- Get system insights")
        print("- Manage agents")
        print("\nCommands: 'exit' to quit, 'history' to see conversation, 'clear' to reset")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("\nðŸ‘¤ You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    print("\nðŸ‘‹ Goodbye! Your conversation has been saved.")
                    break
                
                if user_input.lower() == "history":
                    self._print_conversation_history()
                    continue
                
                if user_input.lower() == "clear":
                    self.conversation_history = []
                    print("âœ“ Conversation cleared.")
                    continue
                
                # Process the message
                print("\nðŸ¤– Assistant: ", end="", flush=True)
                
                response, intent, agent_spawned = self.process_user_message(user_input)
                print(response)
                
                if agent_spawned:
                    print(f"\nâœ… Agent spawned: {agent_spawned}")
                
                if intent:
                    print(f"\n[Intent: {intent.value}]", end="")
                
            except KeyboardInterrupt:
                print("\n\nðŸ‘‹ Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nâœ— Error: {e}")

    def _print_conversation_history(self) -> None:
        """Print the conversation history."""
        
        print("\n" + "="*80)
        print("CONVERSATION HISTORY")
        print("="*80 + "\n")
        
        for msg in self.conversation_history:
            icon = "ðŸ‘¤" if msg.role == "user" else "ðŸ¤–"
            print(f"{icon} {msg.role.upper()}")
            print(f"   {msg.content}")
            if msg.intent:
                print(f"   [Intent: {msg.intent.value}]")
            if msg.agent_spawned:
                print(f"   [Agent spawned: {msg.agent_spawned}]")
            print()
        
        print("="*80 + "\n")

    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation."""
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for m in self.conversation_history if m.role == "user"),
            "assistant_messages": sum(1 for m in self.conversation_history if m.role == "assistant"),
            "agents_spawned": [m.agent_spawned for m in self.conversation_history if m.agent_spawned],
            "intents_detected": [m.intent.value for m in self.conversation_history if m.intent],
            "conversation": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "intent": m.intent.value if m.intent else None,
                    "agent_spawned": m.agent_spawned
                }
                for m in self.conversation_history
            ]
        }

    def save_conversation(self, filename: str = "conversation.json") -> None:
        """Save the conversation to a JSON file."""
        
        summary = self.get_conversation_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"âœ“ Conversation saved to {filename}")

    def load_conversation(self, filename: str = "conversation.json") -> None:
        """Load a previous conversation from a JSON file."""
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.conversation_history = [
                Message(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg["timestamp"],
                    intent=UserIntent[msg["intent"]] if msg["intent"] else None,
                    agent_spawned=msg["agent_spawned"]
                )
                for msg in data["conversation"]
            ]
            
            print(f"âœ“ Conversation loaded from {filename}")
            self._print_conversation_history()
            
        except FileNotFoundError:
            print(f"âœ— File {filename} not found")


# Example usage
if __name__ == "__main__":
    chat = UniversalAIChatInterface()
    chat.multi_turn_conversation()
    chat.save_conversation()
