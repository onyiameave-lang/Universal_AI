import unittest

from shared.mission_manager import MissionManager


class MissionManagerTests(unittest.TestCase):
    def test_create_mission_and_assign_agent(self) -> None:
        manager = MissionManager()
        mission = manager.create_mission(
            mission_id="research",
            description="Investigate new opportunities",
            objectives=["gather evidence", "synthesize findings"],
            domain="research",
        )

        self.assertEqual(mission["id"], "research")
        self.assertEqual(mission["objectives"], ["gather evidence", "synthesize findings"])

        manager.assign_agent("research", "atlas", {"role": "researcher"})
        self.assertIn("atlas", manager.list_agents_for_mission("research"))

        status = manager.get_status()
        self.assertIn("research", status["missions"])
        self.assertEqual(status["missions"]["research"]["agent_count"], 1)


if __name__ == "__main__":
    unittest.main()
