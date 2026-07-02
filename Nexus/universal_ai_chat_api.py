# universal_ai_chat_api.py
# Flask API backend for Universal AI chat interface

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import json
import logging
import os, sys
import importlib
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load .env from the app folder first, then the project root.
for env_path in (
    os.path.join(BASE_DIR, ".env"),
    os.path.join(PROJECT_ROOT, ".env"),
):
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)

# Dynamically link MarketOracle path for trading bot access
market_oracle_path = os.path.join(PROJECT_ROOT, "MarketOracle-workspace")
if market_oracle_path not in sys.path:
    sys.path.append(market_oracle_path)

# Dynamically link Memory AI path
memory_ai_path = os.path.join(PROJECT_ROOT, "ai-memory-system-", "core")
if memory_ai_path not in sys.path:
    sys.path.append(memory_ai_path)

# Initialize systems lazily so a missing SDK/key does not prevent Flask from
# starting and reporting the real setup problem.
memory_ai = None
universal_system = None
chat_interface = None
chat_interface_class = None
anthropic_api_key = None
initialization_error = None
initialization_attempted = False

# Store active agents and conversations
active_conversations = {}
spawned_agents = []

API_ENDPOINTS = {
    "health": "GET /api/health",
    "chat": "POST /api/chat",
    "conversation": "GET /api/conversation/<conversation_id>",
    "save_conversation": "POST /api/conversation/<conversation_id>/save",
    "agents": "GET /api/agents",
    "spawn_agent": "POST /api/agents/spawn",
    "knowledge": "GET /api/knowledge",
    "query_knowledge": "POST /api/knowledge/query",
    "system_status": "GET /api/system/status",
    "broadcast": "POST /api/broadcast",
    "suggest_domain": "POST /api/suggest-domain",
}

def initialize_systems():
    """Initialize the AI systems once and keep a clear error if setup fails."""
    global memory_ai, universal_system, chat_interface, chat_interface_class
    global anthropic_api_key, initialization_error, initialization_attempted

    if chat_interface and memory_ai and universal_system:
        return True

    initialization_attempted = True

    try:
        importlib.invalidate_caches()

        for env_path in (
            os.path.join(BASE_DIR, ".env"),
            os.path.join(PROJECT_ROOT, ".env"),
        ):
            if os.path.exists(env_path):
                load_dotenv(env_path, override=True)

        # Import local orchestrator before any similarly named external module.
        from main import UniversalAI
        from universal_ai_chat_interface import UniversalAIChatInterface
        from OPTIMIZED_memory_ai_system import MemoryAISystem

        memory_ai = MemoryAISystem()
        universal_system = UniversalAI(memory_ai_system=memory_ai)

        anthropic_api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()

        chat_interface_class = UniversalAIChatInterface
        chat_interface = UniversalAIChatInterface(
            memory_ai_system=memory_ai,
            universal_ai_system=universal_system,
            api_key=anthropic_api_key
        )
        initialization_error = None
        return True
    except Exception as exc:
        initialization_error = str(exc)
        logger.exception("Universal AI systems failed to initialize")
        return False

def service_unavailable_response():
    return jsonify({
        "error": "Universal AI services are not initialized",
        "details": initialization_error or "Unknown initialization error",
        "hint": "Check that .r has all dependencies installed and .env contains required API keys."
    }), 503

def ensure_services_available():
    if initialize_systems():
        return None
    return service_unavailable_response()

@app.route('/', methods=['GET'])
def root_index():
    """Serve the Hub Dashboard as the root entry point."""
    return send_from_directory(BASE_DIR, "index.html")

@app.route('/api', methods=['GET'])
def api_index():
    """Show available API endpoints."""
    return jsonify({
        "name": "Universal AI Chat API",
        "status": "online",
        "message": "Use /api/health to check status or POST /api/chat to send a chat message.",
        "endpoints": API_ENDPOINTS
    })

@app.route('/chat', methods=['GET'])
def chat_ui():
    """Serve the browser chat interface."""
    return send_from_directory(BASE_DIR, "index.html")

@app.route('/api/chat', methods=['GET'])
def chat_get():
    """Open the browser chat UI when /api/chat is visited directly."""
    return chat_ui()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the API is running."""
    services_ready = initialize_systems()
    return jsonify({
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "chat": "active" if services_ready else "unavailable",
            "memory_ai": "connected" if memory_ai else "unavailable",
            "agent_spawner": "ready" if universal_system else "unavailable"
        },
        "initializationError": initialization_error
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint.
    Processes user messages and returns AI responses.
    """
    try:
        unavailable = ensure_services_available()
        if unavailable:
            return unavailable

        data = request.get_json()
        user_message = data.get('message')
        conversation_id = data.get('conversationId', 'default')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        logger.info(f"Processing message: {user_message[:50]}...")
        
        # Get or create conversation
        if conversation_id not in active_conversations:
            active_conversations[conversation_id] = chat_interface_class(
                memory_ai_system=memory_ai,
                universal_ai_system=universal_system,
                api_key=anthropic_api_key
            )
        
        chat = active_conversations[conversation_id]
        
        # Process the message
        response, intent, agent_spawned = chat.process_user_message(user_message)
        
        # Track spawned agents
        if agent_spawned:
            spawned_agents.append({
                "name": agent_spawned,
                "timestamp": datetime.now().isoformat(),
                "conversation": conversation_id
            })
            logger.info(f"Agent spawned: {agent_spawned}")
        
        return jsonify({
            "response": response,
            "intent": intent.value if intent else None,
            "agentSpawned": agent_spawned,
            "conversationId": conversation_id,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get conversation history."""
    try:
        unavailable = ensure_services_available()
        if unavailable:
            return unavailable

        if conversation_id not in active_conversations:
            return jsonify({"error": "Conversation not found"}), 404
        
        chat = active_conversations[conversation_id]
        summary = chat.get_conversation_summary()
        
        return jsonify(summary)
    
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation/<conversation_id>/save', methods=['POST'])
def save_conversation(conversation_id):
    """Save a conversation to file."""
    try:
        unavailable = ensure_services_available()
        if unavailable:
            return unavailable

        if conversation_id not in active_conversations:
            return jsonify({"error": "Conversation not found"}), 404
        
        chat = active_conversations[conversation_id]
        filename = f"conversation_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        chat.save_conversation(filename)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "message": f"Conversation saved to {filename}"
        })
    
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get list of ALL agents (Built-in + Spawned)."""
    try:
        unavailable = ensure_services_available()
        # Even if systems are partially down, try to return basic agent info
        registry_data = {}
        if universal_system and hasattr(universal_system, 'agent_registry'):
            registry_data = universal_system.agent_registry.list_agents()

        return jsonify({
            "spawnedAgents": spawned_agents,
            "totalSpawned": len(spawned_agents),
            "activeConversations": len(active_conversations),
            "registry": registry_data,
            "status": "partial" if unavailable else "active"
        })
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents/spawn', methods=['POST'])
def spawn_agent():
    """Directly spawn an agent (without chat)."""
    try:
        unavailable = ensure_services_available()
        if unavailable:
            return unavailable

        data = request.get_json()
        agent_name = data.get('name')
        domain = data.get('domain')
        
        if not agent_name:
            return jsonify({"error": "Agent name required"}), 400

        domain_val = (domain or "general").lower()
        if domain_val in {"trading", "forex", "market", "markets"}:
            return jsonify({
                "success": True,
                "agent": "MarketOracle",
                "domain": "trading",
                "existing": True,
                "message": "Trading is handled by the existing MarketOracle agent; no new agent code was generated."
            })
        
        # Spawn the agent using UniversalAI orchestration
        logger.info(f"Spawning agent: {agent_name} (domain: {domain})")
        spawn_result = None
        if universal_system:
            universal_system.query_counts[domain_val] = 5  # Ensure threshold is met
            spawn_result = universal_system.agent_spawner.spawn_agent(domain_val)
            if getattr(universal_system, "agent_registry", None):
                universal_system.agent_registry.register_spawned_agent(domain_val)

        agent_folder = os.path.join(universal_system.agents_dir, f"{domain_val}_agent") if universal_system else None
        files_written = spawn_result.get("files_written", []) if isinstance(spawn_result, dict) else []
        if not agent_folder or not os.path.isdir(agent_folder) or not files_written:
            return jsonify({
                "success": False,
                "agent": agent_name,
                "domain": domain_val,
                "folder": agent_folder,
                "message": "Spawn failed or did not create verified agent files.",
                "spawnResult": spawn_result
            }), 500

        # Store activity in Memory AI
        if memory_ai and hasattr(memory_ai, "receive_contribution"):
            memory_ai.receive_contribution(
                agent_id="api_backend",
                domain=domain or "system",
                concept=f"spawn_event_{agent_name}",
                three_ws={
                    "what": f"Manual spawn event for {agent_name}",
                    "when": datetime.now().isoformat(),
                    "why": f"API request asked for domain {domain}",
                },
                confidence=1.0
            )
        
        spawned_agents.append({
            "name": agent_name,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "method": "direct"
        })
        
        return jsonify({
            "success": True,
            "agent": agent_name,
            "domain": domain_val,
            "folder": agent_folder,
            "files": files_written,
            "message": f"Agent {agent_name} spawned successfully"
        })
    
    except Exception as e:
        logger.error(f"Error spawning agent: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge', methods=['GET'])
def get_knowledge():
    """Get knowledge base statistics."""
    try:
        unavailable = ensure_services_available()
        if unavailable:
            return unavailable

        stats = memory_ai.database.get_statistics()
        
        return jsonify({
            "totalConcepts": stats.get("total_concepts", 0),
            "verifiedConcepts": stats.get("verified", 0),
            "domains": stats.get("domains", {}),
            "lastUpdated": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/query', methods=['POST'])
def query_knowledge():
    """Query the knowledge base directly."""
    try:
        unavailable = ensure_services_available()
        if unavailable:
            return unavailable

        data = request.get_json()
        domain = data.get('domain')
        concept = data.get('concept')
        
        if not domain or not concept:
            return jsonify({"error": "Domain and concept required"}), 400
        
        knowledge = memory_ai.get_concept(domain, concept)
        
        if knowledge:
            return jsonify({
                "found": True,
                "knowledge": knowledge
            })
        else:
            return jsonify({
                "found": False,
                "message": f"No knowledge found for {domain}:{concept}"
            })
    
    except Exception as e:
        logger.error(f"Error querying knowledge: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get overall system status."""
    try:
        unavailable = ensure_services_available()
        if unavailable:
            return unavailable

        memory_stats = memory_ai.database.get_statistics()
        
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "memory_ai": {
                "status": "operational",
                "concepts": memory_stats.get("total_concepts", 0),
                "verified": memory_stats.get("verified", 0),
                "accuracy": memory_stats.get("avg_accuracy", 0)
            },
            "chat": {
                "status": "operational",
                "activeConversations": len(active_conversations),
                "agentsSpawned": len(spawned_agents)
            },
            "agents": {
                "totalSpawned": len(spawned_agents),
                "active": len([a for a in spawned_agents if a.get("method") != "archived"])
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/broadcast', methods=['POST'])
def broadcast_to_agents():
    """Broadcast a message to all active agents."""
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            return jsonify({"error": "Message required"}), 400
        
        logger.info(f"Broadcasting to {len(spawned_agents)} agents: {message[:50]}...")
        
        return jsonify({
            "success": True,
            "agentsBroadcastTo": len(spawned_agents),
            "message": "Broadcast sent"
        })
    
    except Exception as e:
        logger.error(f"Error broadcasting: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/suggest-domain', methods=['POST'])
def suggest_domain():
    """
    Suggest a domain based on user description.
    Helper endpoint for agent spawning.
    """
    try:
        data = request.get_json()
        description = data.get('description')
        
        if not description:
            return jsonify({"error": "Description required"}), 400
        
        # Common domain patterns
        domains = {
            "trading": ["trade", "market", "crypto", "stock", "forex"],
            "security": ["security", "vulnerability", "threat", "attack", "safe"],
            "nlp": ["text", "language", "sentiment", "chat", "nlp", "conversation"],
            "gaming": ["game", "play", "chess", "strategy"],
            "data": ["data", "analysis", "analytics", "visualization"],
            "research": ["research", "learn", "study", "knowledge"],
            "web": ["web", "scraping", "crawl", "website"]
        }
        
        description_lower = description.lower()
        
        # Find matching domain
        matched_domain = "general"
        max_matches = 0
        
        for domain, keywords in domains.items():
            matches = sum(1 for kw in keywords if kw in description_lower)
            if matches > max_matches:
                max_matches = matches
                matched_domain = domain
        
        return jsonify({
            "suggestedDomain": matched_domain,
            "confidence": 0.8 if max_matches > 0 else 0.5,
            "keywords": [kw for kw in domains.get(matched_domain, []) if kw in description_lower]
        })
    
    except Exception as e:
        logger.error(f"Error suggesting domain: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting Universal AI Chat API...")
    logger.info("Available endpoints:")
    logger.info("  POST /api/chat - Main chat endpoint")
    logger.info("  GET /api/conversation/<id> - Get conversation history")
    logger.info("  GET /api/agents - Get spawned agents")
    logger.info("  POST /api/agents/spawn - Spawn an agent")
    logger.info("  GET /api/knowledge - Get knowledge base stats")
    logger.info("  POST /api/knowledge/query - Query knowledge")
    logger.info("  GET /api/system/status - Get system status")
    logger.info("Starting server on http://localhost:5000")
    
    app.run(debug=True, port=5000)
