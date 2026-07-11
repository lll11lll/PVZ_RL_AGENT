from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pvzrl_env import BridgeTimeoutError, PvZBridgeClient
from pvzrl_adventure import AdventureAttemptLog, _finalize_policy_attempt
from pvzrl_fusion import (
    are_fusion_compatible,
    board_plant_identity_features,
    can_accept_fusion,
    fusion_tier,
)
from pvzrl_sb3 import PvZMaskedPPOEnv


class FusionChainIdentityTests(unittest.TestCase):
    def test_recursive_pea_chain_is_compatible(self) -> None:
        self.assertTrue(are_fusion_compatible(1030, 0))
        self.assertTrue(are_fusion_compatible(1031, 0))
        self.assertTrue(can_accept_fusion(1030))
        self.assertEqual(fusion_tier(1030), 1)
        self.assertEqual(fusion_tier(1031), 2)
        self.assertEqual(fusion_tier(1032), 3)

    def test_board_identity_distinguishes_each_pea_tier(self) -> None:
        identities = {
            board_plant_identity_features({"type": plant_type})
            for plant_type in (0, 1030, 1031, 1032)
        }
        self.assertEqual(len(identities), 4)
        self.assertEqual(board_plant_identity_features({"type": 0}), (0.0, 1.0))

    def test_later_plant_identities_are_distinct(self) -> None:
        identities = {
            board_plant_identity_features({"type": plant_type})
            for plant_type in (1, 3, 2, 4, 6)
        }
        self.assertEqual(len(identities), 5)


class ActionWatchdogTests(unittest.TestCase):
    def test_bridge_timeout_response_uses_timeout_exception(self) -> None:
        client = PvZBridgeClient(timeout=3.0, action_timeout=0.25)
        client.connect = lambda: None  # type: ignore[method-assign]
        client._sock = _FakeSocket()
        client._writer = _FakeWriter()
        client._reader = _FakeReader(json.dumps({"ok": False, "error": "timeout", "details": "main thread stuck"}) + "\n")
        with self.assertRaises(BridgeTimeoutError) as raised:
            client.request("step", action=1)
        self.assertEqual(raised.exception.command, "step")
        self.assertAlmostEqual(raised.exception.timeout, 0.25)

    def test_freeze_record_writes_repro_bundle(self) -> None:
        env = object.__new__(PvZMaskedPPOEnv)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env.config = type(
                "Config",
                (),
                {
                    "enable_action_watchdog": True,
                    "action_diagnostics_path": str(root / "actions.jsonl"),
                    "save_freeze_debug_bundle": True,
                    "freeze_debug_dir": str(root / "freeze"),
                },
            )()
            env._episode_index = 4
            env._step_count = 8
            env._reset_action_diagnostics()
            env.decode_policy_action = lambda action, observation=None: {"kind": "plant", "seed_slot": 1, "row": 2, "col": 3}  # type: ignore[method-assign]
            observation = {
                "currentAdventureLevel": 6,
                "screenState": "gameplay",
                "gameplayReady": True,
                "sun": 100,
                "plants": [],
                "seedSlots": [],
            }
            record = env._record_action_diagnostic(
                policy_action=94,
                bridge_action=94,
                pre_observation=observation,
                post_observation=observation,
                info={"action_result": {"bridgeTimeout": True}},
                started_at=100.0,
                duration=0.5,
                timed_out=True,
                exception_text="timeout",
            )
            self.assertEqual(record["classification"], "action_freeze")
            self.assertTrue(Path(record["debug_bundle_path"]).exists())
            self.assertTrue((root / "actions.jsonl").exists())
            self.assertEqual(env._action_diagnostic_summary()["action_freeze_count"], 1)

    def test_game_over_ui_overrides_stale_win_summary(self) -> None:
        env = type("Env", (), {"_last_observation": {"screenState": "game_over"}})()
        log = AdventureAttemptLog(attempt=1)
        finalized = _finalize_policy_attempt(
            env,  # type: ignore[arg-type]
            log,
            {
                "episode_summary": {"done_reason": "win", "terminal_reason": "win"},
                "raw_observation": {"screenState": "game_over", "restartButtonVisible": True},
                "adventure_state": {"screenState": "game_over", "restartButtonVisible": True},
            },
        )
        self.assertEqual(finalized.result, "loss")
        self.assertEqual(finalized.terminal_reason, "game_over_restart_screen")


class _FakeSocket:
    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def close(self) -> None:
        pass


class _FakeWriter:
    def write(self, value: str) -> None:
        self.value = value

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeReader:
    def __init__(self, value: str):
        self.value = value

    def readline(self) -> str:
        return self.value

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
