from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from pvzrl_streamer_ppo import (
    OffPolicyContaminationError,
    STREAMER_TRANSITION_INFO_KEY,
    STREAMER_TRANSITION_SCHEMA_VERSION,
    StreamerMaskablePPO,
    behavior_transition_from_info,
)
from pvzrl_sb3 import PvZMaskedPPOEnv
from train_ppo import finish_streamer_episode_boundary


def test_source_shutdown_timeout_blocks_phase_handoff_after_base_cleanup() -> None:
    events: list[str] = []

    class Controller:
        def begin_phase(self, phase: str, *, accepting: bool) -> None:
            events.append(f"gate:{phase}:{accepting}")

        def close(self, *, timeout_seconds: float) -> bool:
            events.append(f"source_close:{timeout_seconds}")
            return False

    class Base:
        def close(self) -> None:
            events.append("base_close")

    env = object.__new__(PvZMaskedPPOEnv)
    env.streamer_v1_controller = Controller()
    env.streamer_v1_event_logger = None
    env.base = Base()

    with pytest.raises(RuntimeError, match="streamer_source_shutdown_timeout"):
        env.close()
    assert events == ["gate:STOPPED:False", "source_close:5.0", "base_close"]
    assert env.streamer_v1_controller is None


@dataclass(frozen=True)
class ScriptedStep:
    kind: str
    pre_observation: tuple[float, float]
    post_observation: tuple[float, float]
    reward: float
    mask: tuple[bool, bool, bool] = (True, True, True)
    executed_action: Optional[int] = None
    different_from_policy: bool = False
    execution_succeeded: bool = True
    demo_eligible: bool = False
    terminated: bool = False
    truncated: bool = False
    done_reason: str = ""
    episode_id: str = "episode-1"


class ScriptedStreamerEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, steps: list[ScriptedStep]):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self.steps = list(steps)
        self.index = 0
        self.current_observation = np.asarray(self.steps[0].pre_observation, dtype=np.float32)
        self.final_observation = np.asarray(self.steps[-1].post_observation, dtype=np.float32)
        self.reset_count = 0
        self.calls: list[dict[str, Any]] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self.reset_count += 1
        if self.index < len(self.steps):
            self.current_observation = np.asarray(
                self.steps[self.index].pre_observation,
                dtype=np.float32,
            )
        else:
            self.current_observation = self.final_observation.copy()
        return self.current_observation.copy(), {}

    def action_masks(self) -> np.ndarray:
        if self.index < len(self.steps):
            return np.asarray(self.steps[self.index].mask, dtype=bool)
        return np.ones(3, dtype=bool)

    def step(self, action: int):
        proposed = int(action)
        spec = self.steps[self.index]
        np.testing.assert_allclose(
            self.current_observation,
            np.asarray(spec.pre_observation, dtype=np.float32),
        )
        if spec.kind == "viewer":
            if spec.different_from_policy:
                executed = (proposed + 1) % int(self.action_space.n)
            elif spec.executed_action is not None:
                executed = int(spec.executed_action)
            else:
                executed = proposed
            transition = {
                "schema_version": STREAMER_TRANSITION_SCHEMA_VERSION,
                "behavior_source": "viewer",
                "viewer_controlled": True,
                "proposed_policy_action": proposed,
                "executed_action": executed,
                "execution_succeeded": bool(spec.execution_succeeded),
                "demo_eligible": bool(spec.demo_eligible),
                "execution_status": "success" if spec.execution_succeeded else "bridge_rejected",
                "demonstration": {
                    "episode_id": spec.episode_id,
                    "observation_version": "fixture_v1",
                    "observation_revision": f"frame-{self.index}",
                    "command_type": "slot",
                },
            }
            info: dict[str, Any] = {STREAMER_TRANSITION_INFO_KEY: transition}
        elif spec.kind == "policy":
            executed = proposed
            # Omitting the contract is allowed only for an exact policy step.
            info = {}
        else:
            raise AssertionError(f"unknown scripted step kind={spec.kind!r}")
        if spec.done_reason:
            info["done_reason"] = spec.done_reason
            info["terminal_reason"] = spec.done_reason
            info["episode_summary"] = {
                "episode": spec.episode_id,
                "done_reason": spec.done_reason,
                "result": spec.done_reason,
                "reward_total": float(spec.reward),
                "win": spec.done_reason == "win",
                "loss": spec.done_reason == "loss",
            }

        self.calls.append(
            {
                "kind": spec.kind,
                "pre_observation": self.current_observation.copy(),
                "proposed": proposed,
                "executed": executed,
                "mask": np.asarray(spec.mask, dtype=bool),
            }
        )
        self.index += 1
        self.current_observation = np.asarray(spec.post_observation, dtype=np.float32)
        return (
            self.current_observation.copy(),
            float(spec.reward),
            bool(spec.terminated),
            bool(spec.truncated),
            info,
        )


def _make_model(env: ScriptedStreamerEnv, *, n_steps: int = 2, gamma: float = 0.9, gae_lambda: float = 0.8):
    vec_env = DummyVecEnv([lambda: env])
    model = StreamerMaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=0.0,
        vf_coef=0.0,
        learning_rate=1e-3,
        seed=11,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        bc_enabled=False,
        demonstration_capacity=8,
        verbose=0,
    )
    _total, callback = model._setup_learn(
        total_timesteps=n_steps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="streamer_test",
        progress_bar=False,
    )
    return vec_env, model, callback


def _collect(env: ScriptedStreamerEnv, *, n_steps: int = 2):
    vec_env, model, callback = _make_model(env, n_steps=n_steps)
    assert model.collect_rollouts(
        vec_env,
        callback,
        model.rollout_buffer,
        n_rollout_steps=n_steps,
        use_masking=True,
    )
    return vec_env, model


def _value(model: StreamerMaskablePPO, observation: tuple[float, float]) -> float:
    with th.no_grad():
        tensor = th.as_tensor([observation], dtype=th.float32, device=model.device)
        return float(model.policy.predict_values(tensor).detach().cpu().numpy().reshape(-1)[0])


def test_viewer_transition_never_enters_rollout_and_gae_bootstraps_previewer_state():
    env = ScriptedStreamerEnv(
        [
            ScriptedStep("policy", (0.0, 0.0), (1.0, 0.0), 1.0),
            ScriptedStep(
                "viewer",
                (1.0, 0.0),
                (2.0, 0.0),
                100.0,
                different_from_policy=True,
                demo_eligible=True,
            ),
            ScriptedStep("policy", (2.0, 0.0), (3.0, 0.0), 2.0),
        ]
    )
    vec_env, model = _collect(env)
    try:
        buffer = model.rollout_buffer
        np.testing.assert_allclose(buffer.observations[:, 0, 0], [0.0, 2.0])
        assert model.num_timesteps == 2
        assert model.total_environment_actions == 3
        assert model.viewer_interventions == 1
        assert model.last_rollout_policy_transitions == 2
        assert model.last_rollout_viewer_transitions == 1
        assert len(model.demonstration_buffer) == 1

        assert len(model.last_rollout_boundaries) == 1
        boundary = model.last_rollout_boundaries[0]
        assert boundary.rollout_index == 0
        assert boundary.pre_viewer_value == pytest.approx(_value(model, (1.0, 0.0)))
        assert float(buffer.rewards[0, 0]) == pytest.approx(
            1.0 + model.gamma * boundary.pre_viewer_value
        )
        # The first post-viewer policy sample starts a new GAE segment.
        assert bool(buffer.episode_starts[1, 0]) is True
        expected_first_advantage = float(buffer.rewards[0, 0] - buffer.values[0, 0])
        assert float(buffer.advantages[0, 0]) == pytest.approx(expected_first_advantage, abs=1e-5)
        assert float(buffer.rewards[0, 0]) != pytest.approx(101.0)

        # Every stored old log probability belongs to its stored policy action
        # under the exact mask captured at that observation.
        observations = th.as_tensor(
            buffer.observations[:, 0], dtype=th.float32, device=model.device
        )
        actions = th.as_tensor(
            buffer.actions[:, 0, 0], dtype=th.long, device=model.device
        )
        with th.no_grad():
            _values, log_prob, _entropy = model.policy.evaluate_actions(
                observations,
                actions,
                action_masks=buffer.action_masks[:, 0],
            )
        np.testing.assert_allclose(
            log_prob.cpu().numpy(),
            buffer.log_probs[:, 0],
            atol=1e-6,
        )
    finally:
        vec_env.close()


def test_back_to_back_viewer_steps_cut_once_and_same_action_rejection_is_excluded():
    env = ScriptedStreamerEnv(
        [
            ScriptedStep("policy", (0.0, 0.0), (1.0, 0.0), 1.5),
            ScriptedStep(
                "viewer",
                (1.0, 0.0),
                (2.0, 0.0),
                50.0,
                different_from_policy=True,
                demo_eligible=True,
            ),
            ScriptedStep(
                "viewer",
                (2.0, 0.0),
                (3.0, 0.0),
                60.0,
                # No fixed action means executed == proposed.  It is still a
                # viewer transition and is rejected as a positive demo.
                execution_succeeded=False,
                demo_eligible=False,
            ),
            ScriptedStep("policy", (3.0, 0.0), (4.0, 0.0), 2.5),
        ]
    )
    vec_env, model = _collect(env)
    try:
        buffer = model.rollout_buffer
        np.testing.assert_allclose(buffer.observations[:, 0, 0], [0.0, 3.0])
        assert model.total_environment_actions == 4
        assert model.num_timesteps == 2
        assert model.viewer_interventions == 2
        assert len(model.last_rollout_boundaries) == 1
        assert len(model.demonstration_buffer) == 1
        boundary = model.last_rollout_boundaries[0]
        assert float(buffer.rewards[0, 0]) == pytest.approx(
            1.5 + model.gamma * _value(model, (1.0, 0.0))
        )
        assert boundary.reward_after == pytest.approx(float(buffer.rewards[0, 0]))
        assert env.calls[2]["proposed"] == env.calls[2]["executed"]
        assert bool(buffer.episode_starts[1, 0]) is True
    finally:
        vec_env.close()


def test_terminal_viewer_action_resets_before_policy_resumes_and_preserves_execution_mask():
    viewer_mask = (False, True, False)
    reset_policy_mask = (True, False, True)
    env = ScriptedStreamerEnv(
        [
            ScriptedStep("policy", (0.0, 0.0), (1.0, 0.0), 1.0),
            ScriptedStep(
                "viewer",
                (1.0, 0.0),
                (99.0, 99.0),
                10.0,
                mask=viewer_mask,
                executed_action=1,
                demo_eligible=True,
                terminated=True,
                done_reason="win",
                episode_id="terminal-viewer",
            ),
            ScriptedStep(
                "policy",
                (10.0, 0.0),
                (11.0, 0.0),
                2.0,
                mask=reset_policy_mask,
            ),
        ]
    )
    vec_env, model = _collect(env)
    try:
        buffer = model.rollout_buffer
        np.testing.assert_allclose(buffer.observations[:, 0, 0], [0.0, 10.0])
        np.testing.assert_array_equal(buffer.action_masks[1, 0], reset_policy_mask)
        assert env.reset_count == 2  # initial reset plus DummyVecEnv terminal reset
        assert bool(buffer.episode_starts[1, 0]) is True
        records = model.demonstration_buffer.records()
        assert len(records) == 1
        np.testing.assert_array_equal(records[0].action_mask, viewer_mask)
        assert records[0].action == 1
        assert records[0].metadata["episode_outcome"] == "win"
        assert records[0].metadata["episode_outcome_metadata"]["terminal_reason"] == "win"
    finally:
        vec_env.close()


def test_policy_action_mismatch_and_legacy_override_fail_closed():
    with pytest.raises(OffPolicyContaminationError, match="different action"):
        behavior_transition_from_info(
            {
                STREAMER_TRANSITION_INFO_KEY: {
                    "schema_version": 1,
                    "behavior_source": "policy",
                    "viewer_controlled": False,
                    "proposed_policy_action": 0,
                    "executed_action": 1,
                    "execution_succeeded": True,
                    "demo_eligible": False,
                }
            },
            0,
        )
    with pytest.raises(OffPolicyContaminationError, match="without explicit"):
        behavior_transition_from_info(
            {
                "human_coach": {
                    "source": "stream",
                    "event": "coach_match",
                    "command": {"kind": "plant"},
                    "override_applied": False,
                }
            },
            0,
        )
    with pytest.raises(OffPolicyContaminationError, match="without explicit"):
        behavior_transition_from_info(
            {
                "action_source": "TWITCH",
                "viewer_action": 0,
            },
            0,
        )
    with pytest.raises(OffPolicyContaminationError, match="without explicit"):
        behavior_transition_from_info(
            {
                "action_source": "MODEL",
                "action_result": {"executedAction": 2},
            },
            0,
        )


def test_phase_handoff_finishes_partial_episode_without_advancing_policy_steps() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self._last_episode_starts = np.asarray([False])
            self._last_obs = np.asarray([[0.0, 0.0]], dtype=np.float32)
            self.num_timesteps = 25_000
            self.predict_calls = 0

        def predict(self, observation: Any, *, deterministic: bool, action_masks: Any):
            assert deterministic is True
            assert np.asarray(action_masks).shape == (1, 3)
            self.predict_calls += 1
            return np.asarray([1]), None

    class FakeVecEnv:
        def __init__(self) -> None:
            self.steps = 0

        def step(self, _actions: Any):
            self.steps += 1
            return (
                np.asarray([[float(self.steps), 0.0]], dtype=np.float32),
                np.asarray([0.0]),
                np.asarray([self.steps == 3]),
                [{}],
            )

    model = FakeModel()
    vec_env = FakeVecEnv()
    actions = finish_streamer_episode_boundary(
        model,
        vec_env,
        maximum_actions=5,
        action_masks_fn=lambda _env: np.asarray([[True, True, True]]),
    )
    assert actions == 3
    assert model.num_timesteps == 25_000
    assert model.predict_calls == 3
    assert model._last_episode_starts.tolist() == [True]
