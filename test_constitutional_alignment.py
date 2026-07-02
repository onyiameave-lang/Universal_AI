import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "universal-ai"))


def test_constitutional_roles_are_exposed():
    from core.agent_registry import get_constitutional_agent_definitions

    definitions = get_constitutional_agent_definitions()
    expected_roles = {
        "chronicle",
        "oracle",
        "atlas",
        "sentinel",
        "pulse",
        "genesis",
        "forge",
        "nexus",
        "aegis",
    }

    assert expected_roles.issubset(definitions.keys())
    for role, meta in definitions.items():
        assert meta["repository"], f"{role} should declare a repository"
        assert meta["mission"]["purpose"], f"{role} should have a mission purpose"
        assert meta["capabilities"], f"{role} should expose capabilities"
