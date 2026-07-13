"""Bridge-free tests for the shared fusion reward policy."""

from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from pvzrl_env import (
    EpisodeLog,
    PvZEnvConfig,
    PvZGymEnv,
    RewardConfig,
    accumulate_reward_episode_totals,
)
from pvzrl_fusion import (
    apply_fusion_attempt_result,
    default_fusion_diagnostics,
    fusion_live_fields,
    merge_episode_fusion_stats,
)
from pvzrl_rewards import fusion_source_from_result
from train_ppo import (
    EPISODE_METRIC_FIELDS,
    PROGRESS_CSV_DIAGNOSTIC_FIELDS,
    build_reward_config,
    clean_episode_row,
    summarize_episode_rows,
    write_progress_csv_rows,
)


REWARD_FIELDS = (
    "fusion_reward_total",
    "fusion_attempt_reward_total",
    "fusion_success_reward_total",
    "fusion_threatened_row_bonus_total",
    "fusion_active_wave_bonus_total",
    "fusion_defensive_value_bonus_total",
    "fusion_incompatible_penalty_total",
    "fusion_empty_tile_penalty_total",
    "fusion_failed_penalty_total",
    "fusion_bridge_error_penalty_total",
    "fusion_spam_penalty_total",
    "fusion_reward_capped",
    "fusion_last_reward_delta",
    "fusion_last_reward_reason",
    "fusion_last_usefulness_bonus",
)


def observation(*, threatened: bool = False) -> dict:
    return {
        "rowCount": 5,
        "columnCount": 10,
        "gameplayReady": True,
        "boardFound": True,
        "sun": 500,
        "seedSlots": [
            {"slotIndex": 0, "plantType": 1, "plantTypeName": "SunFlower", "ready": True, "usable": True},
            {"slotIndex": 1, "plantType": 0, "plantTypeName": "Peashooter", "ready": True, "usable": True},
        ],
        "plants": [{"row": 2, "column": 4, "type": 1, "typeName": "SunFlower"}],
        "zombies": ([{"row": 2, "alive": True}] if threatened else []),
        "lanes": ([{"row": 2, "zombieCount": 1, "danger": 0.8}] if threatened else []),
    }


def candidate(*, legal: bool = True, col: int = 4) -> dict:
    return {
        "source_plant_type": 1,
        "source_plant_name": "SunFlower",
        "source_row": 2,
        "source_col": col,
        "target_or_ingredient_type": 0,
        "target_or_ingredient_name": "Peashooter",
        "ingredient_seed_slot_index": 1,
        "fusion_legal": legal,
        "fusion_blocked_reason": "" if legal else "incompatible_pair",
    }


def action_result(*, success: bool, reason: str = "", legal: bool = True, col: int = 4) -> dict:
    item = candidate(legal=legal, col=col)
    return {
        "fusionAttempted": True,
        "fusionSucceeded": success,
        "fusion_success": success,
        "fusionRejectedReason": reason,
        "illegalReason": reason or None,
        "changedTileCount": 1 if success else 0,
        "bridgeMethodUsed": "test_bridge",
        "fusionExecutionSource": "model_action_mask",
        "fusionCandidate": item,
        "decoded": {
            "kind": "fusion",
            "sourcePlantType": 1,
            "ingredientPlantType": 0,
            "row": 2,
            "column": col,
        },
    }


class FusionRewardPolicyTests(unittest.TestCase):
    def make_env(self, reward: RewardConfig | None = None) -> PvZGymEnv:
        return PvZGymEnv(PvZEnvConfig(fusion_policy="observe", reward=reward or RewardConfig()))

    def apply_event(self, env: PvZGymEnv, result: dict, obs: dict, rejected_reason: str = "") -> dict:
        item = result["fusionCandidate"]
        diag = apply_fusion_attempt_result(
            default_fusion_diagnostics("observe"),
            item,
            result,
            rejected_reason=rejected_reason,
        )
        delta = env._compose_step_reward(
            obs,
            obs,
            result,
            previous_legal_actions=[],
        ).breakdown.component("fusion_reward")
        diag.update(env._fusion_reward_live_fields())
        return {"delta": delta, "diagnostics": diag}

    def test_successful_legal_fusion_is_rewarded(self) -> None:
        env = self.make_env()
        outcome = self.apply_event(env, action_result(success=True), observation())
        self.assertGreater(outcome["delta"], 0.0)
        self.assertEqual(outcome["diagnostics"]["fusion_success_count"], 1)
        self.assertGreater(outcome["diagnostics"]["fusion_success_reward_total"], 0.0)
        self.assertGreater(outcome["diagnostics"]["fusion_attempt_reward_total"], 0.0)

    def test_incompatible_and_empty_fusions_get_no_success_reward(self) -> None:
        for reason, field in (
            ("incompatible_pair", "fusion_incompatible_penalty_total"),
            ("empty_tile", "fusion_empty_tile_penalty_total"),
        ):
            with self.subTest(reason=reason):
                env = self.make_env()
                result = action_result(success=False, reason=reason, legal=False)
                outcome = self.apply_event(env, result, observation(), rejected_reason=reason)
                self.assertLessEqual(outcome["delta"], 0.0)
                self.assertEqual(outcome["diagnostics"]["fusion_success_reward_total"], 0.0)
                self.assertLess(outcome["diagnostics"][field], 0.0)
                self.assertEqual(outcome["diagnostics"]["fusion_rejected_count"], 1)

    def test_threatened_row_receives_contextual_bonuses(self) -> None:
        env = self.make_env()
        outcome = self.apply_event(env, action_result(success=True), observation(threatened=True))
        diag = outcome["diagnostics"]
        self.assertGreater(diag["fusion_threatened_row_bonus_total"], 0.0)
        self.assertGreater(diag["fusion_active_wave_bonus_total"], 0.0)
        self.assertGreater(diag["fusion_defensive_value_bonus_total"], 0.0)

    def test_positive_cap_does_not_block_penalties(self) -> None:
        reward = RewardConfig(max_fusion_reward_per_episode=0.60)
        env = self.make_env(reward)
        first = self.apply_event(env, action_result(success=True, col=4), observation())
        second = self.apply_event(env, action_result(success=True, col=5), observation())
        self.assertAlmostEqual(env._reward_state.fusion.positive_total, 0.60, places=7)
        self.assertAlmostEqual(env._reward_state.fusion.reward_total, 0.60, places=7)
        self.assertTrue(second["diagnostics"]["fusion_reward_capped"])

        bad = action_result(success=False, reason="bridge_error", legal=True, col=6)
        penalty = self.apply_event(env, bad, observation(), rejected_reason="bridge_error")
        self.assertLess(penalty["delta"], 0.0)
        self.assertLess(env._reward_state.fusion.reward_total, 0.60)
        totals = env._fusion_reward_live_fields()
        component_sum = sum(
            float(totals[field])
            for field in REWARD_FIELDS
            if field.endswith("_total") and field != "fusion_reward_total"
        )
        self.assertAlmostEqual(component_sum, totals["fusion_reward_total"], places=7)

    def test_repeated_bad_attempt_gets_spam_penalty(self) -> None:
        env = self.make_env()
        result = action_result(success=False, reason="incompatible_pair", legal=False)
        first = self.apply_event(env, result, observation(), rejected_reason="incompatible_pair")
        second = self.apply_event(env, result, observation(), rejected_reason="incompatible_pair")
        self.assertEqual(first["diagnostics"]["fusion_spam_penalty_total"], 0.0)
        self.assertLess(second["diagnostics"]["fusion_spam_penalty_total"], 0.0)
        self.assertIn("spam", second["diagnostics"]["fusion_last_reward_reason"])

    def test_reset_clears_reward_cap_and_spam_history(self) -> None:
        env = self.make_env(RewardConfig(max_fusion_reward_per_episode=0.1))
        self.apply_event(env, action_result(success=True), observation())
        self.apply_event(
            env,
            action_result(success=False, reason="incompatible_pair", legal=False),
            observation(),
            rejected_reason="incompatible_pair",
        )
        env._reset_reward_episode_state()
        self.assertEqual(env._reward_state.fusion.reward_total, 0.0)
        self.assertEqual(env._reward_state.fusion.positive_total, 0.0)
        self.assertFalse(env._reward_state.fusion.capped)
        self.assertEqual(len(env._reward_state.fusion.recent_attempts), 0)

    def test_live_and_episode_metrics_include_reward_fields(self) -> None:
        env = self.make_env()
        outcome = self.apply_event(env, action_result(success=True), observation())
        live = fusion_live_fields(outcome["diagnostics"], "observe")
        for field in REWARD_FIELDS:
            self.assertIn(field, live)

        log = EpisodeLog(policy="ppo", episode_index=1)
        merge_episode_fusion_stats(log, outcome["diagnostics"])
        accumulate_reward_episode_totals(
            log,
            {"reward_breakdown": {"fusion_reward": outcome["delta"]}},
        )
        payload = asdict(log)
        for field in REWARD_FIELDS:
            self.assertIn(field, payload)
        self.assertGreater(payload["fusion_reward_total"], 0.0)
        self.assertGreater(payload["fusion_success_reward_total"], 0.0)

        row = clean_episode_row(
            {
                "episode": 1,
                "episode_reward": outcome["delta"],
                "fusion_reward_total": outcome["delta"],
                **env._fusion_reward_live_fields(),
            },
            1,
        )
        summary = summarize_episode_rows([row], 1, Path("."), Path("model.zip"), 1.0)
        self.assertGreater(summary["fusion_reward_total"], 0.0)
        self.assertGreater(summary["fusion_success_reward_total"], 0.0)

    def test_progress_csv_writer_accepts_new_diagnostics_with_old_header(self) -> None:
        old_fieldnames = [field for field in EPISODE_METRIC_FIELDS if field not in PROGRESS_CSV_DIAGNOSTIC_FIELDS]
        row = clean_episode_row(
            {
                "episode": 7,
                "episode_reward": 1.5,
                "recursive_fusion_count": 2,
                "highest_fusion_tier": 3,
                "action_freeze_count": 1,
                "mean_action_duration_seconds": 0.25,
                "p95_action_duration_seconds": 0.5,
                "max_action_duration_seconds": 0.75,
                "unexpected_future_field": "ignored",
            },
            7,
        )

        with tempfile.TemporaryDirectory(prefix="pvzrl_progress_csv_test_") as temp_dir:
            csv_path = Path(temp_dir) / "episode_metrics.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=old_fieldnames)
                writer.writeheader()

            header_written, writer_fieldnames = write_progress_csv_rows(csv_path, [row], old_fieldnames, True)
            self.assertTrue(header_written)
            for field in PROGRESS_CSV_DIAGNOSTIC_FIELDS:
                self.assertIn(field, writer_fieldnames)

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertIsNotNone(reader.fieldnames)
                for field in PROGRESS_CSV_DIAGNOSTIC_FIELDS:
                    self.assertIn(field, reader.fieldnames or [])
                rows = list(reader)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recursive_fusion_count"], "2")
        self.assertEqual(rows[0]["highest_fusion_tier"], "3")
        self.assertEqual(rows[0]["action_freeze_count"], "1")
        self.assertEqual(rows[0]["mean_action_duration_seconds"], "0.25")
        self.assertEqual(rows[0]["p95_action_duration_seconds"], "0.5")
        self.assertEqual(rows[0]["max_action_duration_seconds"], "0.75")

    def test_reward_cli_mapping_uses_tuning_values(self) -> None:
        args = argparse.Namespace(
            fusion_attempt_reward=0.03,
            fusion_success_reward=0.7,
            max_fusion_reward_per_episode=1.25,
        )
        reward = build_reward_config(args, {})
        self.assertEqual(reward["fusion_attempt_reward"], 0.03)
        self.assertEqual(reward["fusion_success_reward"], 0.7)
        self.assertEqual(reward["max_fusion_reward_per_episode"], 1.25)

    def test_source_classification_covers_all_shared_paths(self) -> None:
        env = self.make_env()
        cases = (
            ({"fusionExecutionSource": "model_action_mask"}, "model"),
            ({"fusionExecutionSource": "scripted"}, "scripted"),
            ({"humanCoach": {"source": "human", "commandMode": "override"}}, "human_coach"),
            ({"humanCoach": {"source": "stream"}}, "stream_coach"),
            ({"humanCoach": {"source": "human", "commandMode": "assist"}}, "assist"),
            ({"fusionIntentSource": "gui", "coach_command_source": "human"}, "gui"),
            ({"fusionIntentSource": "manual"}, "manual"),
            ({"fusionIntentSource": "debug"}, "debug"),
        )
        for result, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(fusion_source_from_result(result), expected)

    def test_model_bridge_error_is_returned_to_shared_reward_path(self) -> None:
        class FailingClient:
            def request(self, command: str, **_kwargs):
                self.command = command
                raise RuntimeError("bridge unavailable")

        env = PvZGymEnv(
            PvZEnvConfig(
                plant_types=[1, 0],
                fusion_policy="observe",
                fusion_action_mask_enabled=True,
            )
        )
        env.client = FailingClient()
        result, diagnostics = env._maybe_execute_model_fusion(
            observation(),
            default_fusion_diagnostics("observe"),
            requested_action=75,
            executed_action=75,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["fusionAttempted"])
        self.assertEqual(result["fusionRejectedReason"], "bridge_error")
        self.assertEqual(diagnostics["fusion_bridge_error_count"], 1)
        self.assertEqual(diagnostics["fusion_failed_count"], 1)
        delta = env._compose_step_reward(
            observation(),
            observation(),
            result,
            previous_legal_actions=[],
        ).breakdown.component("fusion_reward")
        self.assertLess(delta, 0.0)
        self.assertLess(env._fusion_reward_live_fields()["fusion_bridge_error_penalty_total"], 0.0)

    def test_generalist_episode_aggregation_uses_canonical_row_reducers(self) -> None:
        first = {
            "threat_steps_by_row": {"10": 2, "2": 3, "alpha": 1},
            "undefended_threat_age_avg_by_row": {"2": 2.0, "alpha": 5.0},
            "undefended_threat_age_max_by_row": {"2": 7, "alpha": 5},
            "threatened_rows_with_zero_defender_steps_by_row": {"2": 2, "alpha": 1},
            "first_defense_step_by_row": {"2": 5, "alpha": 0},
            "all_rows_peashooter_covered_step": 12,
        }
        second = {
            "threat_steps_by_row": {"2": 4, "alpha": 2, "bad": "not-a-count"},
            "undefended_threat_age_avg_by_row": {"2": 8.0, "alpha": 1.0},
            "undefended_threat_age_max_by_row": {"2": 4, "10": 9},
            "threatened_rows_with_zero_defender_steps_by_row": {"2": 1, "alpha": 3},
            "first_defense_step_by_row": {"2": 7, "10": 3, "alpha": -1},
        }

        summary = summarize_episode_rows([first, second], 2, Path("."), Path("model.zip"), 1.0)

        self.assertEqual(summary["threat_steps_by_row"], {"2": 7, "10": 2, "alpha": 3})
        self.assertEqual(summary["undefended_threat_age_max_by_row"], {"2": 7, "10": 9, "alpha": 5})
        self.assertEqual(summary["undefended_threat_age_avg_by_row"], {"2": 4.0, "alpha": 2.0})
        self.assertEqual(summary["first_defense_step_by_row"], {"2": 6.0, "10": 3.0})
        self.assertEqual(summary["all_rows_peashooter_covered_step"], 12.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
