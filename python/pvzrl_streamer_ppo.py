"""Intervention-aware MaskablePPO and masked behavior cloning for Streamer V1.

This module deliberately owns the RL boundary.  Viewer actions may mutate the
game, but they are never inserted into the on-policy rollout buffer.  A caller
must classify each environment step with the explicit ``streamer_transition``
info contract documented by :class:`BehaviorTransition`.

The implementation targets the maintained single-environment PvZRL trainer.
Supporting partially intervened vector environments would require a per-env
valid-sample buffer and is intentionally rejected instead of approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

from pvzrl_demonstrations import (
    DemonstrationBuffer,
    DemonstrationValidationError,
)


STREAMER_TRANSITION_INFO_KEY = "streamer_transition"
STREAMER_TRANSITION_SCHEMA_VERSION = 1
BEHAVIOR_SOURCE_POLICY = "policy"
BEHAVIOR_SOURCE_VIEWER = "viewer"


class OffPolicyContaminationError(RuntimeError):
    """Raised before a mismatched behavior transition can enter PPO."""


class TransitionClassifier(Protocol):
    """Injectable adapter from environment info to the RL behavior contract."""

    def __call__(
        self,
        info: Mapping[str, Any],
        proposed_policy_action: int,
    ) -> "BehaviorTransition | Mapping[str, Any]": ...


@dataclass(frozen=True, slots=True)
class BehaviorTransition:
    """Authoritative classification of one actual environment transition.

    Environment/controller integration contract (stored under
    ``info["streamer_transition"]`` by default)::

        {
          "schema_version": 1,
          "behavior_source": "policy" | "viewer",
          "viewer_controlled": bool,
          "proposed_policy_action": int,
          "executed_action": int,
          "execution_succeeded": bool,
          "demo_eligible": bool,
          "execution_status": str,
          "demonstration": {... JSON-safe metadata ...}
        }

    ``viewer_controlled`` means Twitch/another viewer source owned the step. It
    remains true when the viewer chose the same action as PPO and when the
    bridge rejected the attempted action.  Those transitions are always
    excluded from PPO.  Only a successful transition with
    ``demo_eligible=true`` becomes a positive behavior-cloning example.
    """

    behavior_source: str
    viewer_controlled: bool
    proposed_policy_action: int
    executed_action: int
    execution_succeeded: bool
    demo_eligible: bool
    execution_status: str = ""
    demonstration: dict[str, Any] = field(default_factory=dict)
    schema_version: int = STREAMER_TRANSITION_SCHEMA_VERSION

    @classmethod
    def policy(cls, proposed_policy_action: int) -> "BehaviorTransition":
        action = int(proposed_policy_action)
        return cls(
            behavior_source=BEHAVIOR_SOURCE_POLICY,
            viewer_controlled=False,
            proposed_policy_action=action,
            executed_action=action,
            execution_succeeded=True,
            demo_eligible=False,
            execution_status="policy_executed",
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_proposed_action: int,
    ) -> "BehaviorTransition":
        if not isinstance(payload, Mapping):
            raise OffPolicyContaminationError("streamer transition contract must be an object")
        required = {
            "schema_version",
            "behavior_source",
            "viewer_controlled",
            "proposed_policy_action",
            "executed_action",
            "execution_succeeded",
            "demo_eligible",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise OffPolicyContaminationError(
                f"streamer transition contract is missing fields: {','.join(missing)}"
            )
        schema_version = int(payload["schema_version"])
        if schema_version != STREAMER_TRANSITION_SCHEMA_VERSION:
            raise OffPolicyContaminationError(
                f"unsupported streamer transition schema_version={schema_version}"
            )
        if type(payload["viewer_controlled"]) is not bool:  # noqa: E721 - exact bool is intentional
            raise OffPolicyContaminationError("viewer_controlled must be a boolean")
        if type(payload["execution_succeeded"]) is not bool:  # noqa: E721
            raise OffPolicyContaminationError("execution_succeeded must be a boolean")
        if type(payload["demo_eligible"]) is not bool:  # noqa: E721
            raise OffPolicyContaminationError("demo_eligible must be a boolean")

        source = str(payload["behavior_source"] or "").strip().lower()
        if source not in {BEHAVIOR_SOURCE_POLICY, BEHAVIOR_SOURCE_VIEWER}:
            raise OffPolicyContaminationError(f"unknown behavior_source={source!r}")
        viewer_controlled = bool(payload["viewer_controlled"])
        if viewer_controlled != (source == BEHAVIOR_SOURCE_VIEWER):
            raise OffPolicyContaminationError(
                "viewer_controlled does not agree with behavior_source"
            )
        proposed = int(payload["proposed_policy_action"])
        if proposed != int(expected_proposed_action):
            raise OffPolicyContaminationError(
                "collector/environment proposed-action mismatch: "
                f"collector={int(expected_proposed_action)} info={proposed}"
            )
        executed = int(payload["executed_action"])
        succeeded = bool(payload["execution_succeeded"])
        demo_eligible = bool(payload["demo_eligible"])
        if source == BEHAVIOR_SOURCE_POLICY and executed != proposed:
            raise OffPolicyContaminationError(
                "policy transition executed a different action: "
                f"proposed={proposed} executed={executed}"
            )
        if source == BEHAVIOR_SOURCE_POLICY and demo_eligible:
            raise OffPolicyContaminationError("policy transition cannot be demo_eligible")
        if demo_eligible and not succeeded:
            raise OffPolicyContaminationError(
                "demo_eligible viewer transition must declare execution_succeeded=true"
            )
        demonstration = payload.get("demonstration") or {}
        if not isinstance(demonstration, Mapping):
            raise OffPolicyContaminationError("demonstration metadata must be an object")
        return cls(
            behavior_source=source,
            viewer_controlled=viewer_controlled,
            proposed_policy_action=proposed,
            executed_action=executed,
            execution_succeeded=succeeded,
            demo_eligible=demo_eligible,
            execution_status=str(payload.get("execution_status") or ""),
            demonstration=dict(demonstration),
            schema_version=schema_version,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "behavior_source": str(self.behavior_source),
            "viewer_controlled": bool(self.viewer_controlled),
            "proposed_policy_action": int(self.proposed_policy_action),
            "executed_action": int(self.executed_action),
            "execution_succeeded": bool(self.execution_succeeded),
            "demo_eligible": bool(self.demo_eligible),
            "execution_status": str(self.execution_status),
            "demonstration": dict(self.demonstration),
        }


@dataclass(frozen=True, slots=True)
class InterventionBoundary:
    """Diagnostic proof of a pre-viewer value bootstrap."""

    rollout_index: int
    pre_viewer_value: float
    gamma: float
    reward_before: float
    reward_after: float


def _legacy_intervention_signal(
    info: Mapping[str, Any],
    proposed_policy_action: Optional[int] = None,
) -> bool:
    """Fail closed if legacy override info appears without the new contract."""

    action_source = str(info.get("action_source") or "").strip().lower()
    if action_source in {
        "crowd",
        "human",
        "human_coach",
        "mock",
        "mock_stream",
        "stream",
        "stream_coach",
        "twitch",
        "viewer",
    }:
        return True
    if info.get("viewer_action") is not None:
        return True

    action_result = info.get("action_result")
    if proposed_policy_action is not None and isinstance(action_result, Mapping):
        executed = action_result.get("executedAction")
        if executed is not None:
            try:
                if int(executed) != int(proposed_policy_action):
                    return True
            except (TypeError, ValueError):
                return True

    for key in ("human_coach", "stream_coach"):
        payload = info.get(key)
        if not isinstance(payload, Mapping):
            continue
        if payload.get("override_applied") is True:
            return True
        source = str(payload.get("source") or "").strip().lower()
        event = str(payload.get("event") or "").strip().lower()
        if source in {"stream", "viewer", "twitch", "human"} and event in {
            "coach_override",
            "coach_match",
            "executed",
        } and payload.get("command") is not None:
            return True
    return False


def behavior_transition_from_info(
    info: Mapping[str, Any],
    proposed_policy_action: int,
    *,
    info_key: str = STREAMER_TRANSITION_INFO_KEY,
) -> BehaviorTransition:
    payload = info.get(info_key)
    if payload is None:
        if _legacy_intervention_signal(info, proposed_policy_action):
            raise OffPolicyContaminationError(
                "legacy coach/viewer override detected without explicit streamer_transition contract"
            )
        return BehaviorTransition.policy(proposed_policy_action)
    if isinstance(payload, BehaviorTransition):
        transition = payload
        if transition.proposed_policy_action != int(proposed_policy_action):
            raise OffPolicyContaminationError(
                "collector/environment proposed-action mismatch: "
                f"collector={int(proposed_policy_action)} info={transition.proposed_policy_action}"
            )
        return transition
    if not isinstance(payload, Mapping):
        raise OffPolicyContaminationError("streamer_transition info must be an object")
    return BehaviorTransition.from_mapping(
        payload,
        expected_proposed_action=int(proposed_policy_action),
    )


SelfStreamerMaskablePPO = TypeVar("SelfStreamerMaskablePPO", bound="StreamerMaskablePPO")


class StreamerMaskablePPO(MaskablePPO):
    """MaskablePPO whose timestep unit is a policy-generated transition."""

    def __init__(
        self,
        *args: Any,
        demonstration_buffer: Optional[DemonstrationBuffer] = None,
        demonstration_capacity: int = 2048,
        demonstration_persist_path: Optional[Path | str] = None,
        demonstration_persist_every: int = 512,
        transition_classifier: Optional[TransitionClassifier] = None,
        transition_info_key: str = STREAMER_TRANSITION_INFO_KEY,
        bc_enabled: bool = True,
        bc_coefficient: float = 0.01,
        bc_batch_size: int = 32,
        bc_update_frequency: int = 1,
        bc_min_demonstrations: int = 32,
        bc_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transition_info_key = str(transition_info_key or STREAMER_TRANSITION_INFO_KEY)
        self._transition_classifier = transition_classifier
        self._streamer_checkpoint_hook: Optional[Any] = None
        self.demonstration_persist_every = max(1, int(demonstration_persist_every))
        if demonstration_buffer is None:
            observation_shape: Optional[Sequence[int]] = None
            action_count: Optional[int] = None
            observation_space = getattr(self, "observation_space", None)
            action_space = getattr(self, "action_space", None)
            if isinstance(observation_space, spaces.Box):
                observation_shape = tuple(int(value) for value in observation_space.shape)
            if isinstance(action_space, spaces.Discrete):
                action_count = int(action_space.n)
            demonstration_buffer = DemonstrationBuffer(
                max(1, int(demonstration_capacity)),
                observation_shape=observation_shape,
                action_count=action_count,
                persist_path=demonstration_persist_path,
            )
        self.demonstration_buffer = demonstration_buffer

        self.bc_enabled = bool(bc_enabled)
        self.bc_coefficient = float(bc_coefficient)
        if not np.isfinite(self.bc_coefficient) or not 0.0 <= self.bc_coefficient <= 1.0:
            raise ValueError("bc_coefficient must be finite and in [0, 1]")
        self.bc_batch_size = max(1, int(bc_batch_size))
        self.bc_update_frequency = max(1, int(bc_update_frequency))
        self.bc_min_demonstrations = max(1, int(bc_min_demonstrations))
        effective_seed = int(bc_seed) if bc_seed is not None else int(getattr(self, "seed", 0) or 0) + 947
        self.bc_seed = effective_seed
        self._bc_rng = np.random.default_rng(effective_seed)

        self.total_environment_actions = 0
        self.viewer_interventions = 0
        self.policy_transitions_collected = 0
        self.demonstrations_recorded = int(len(self.demonstration_buffer))
        self.bc_update_count = 0
        self._streamer_train_calls = 0
        self.last_bc_loss = 0.0
        self.last_bc_policy_agreement = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy_loss = 0.0
        self.last_rollout_boundaries: list[InterventionBoundary] = []
        self.last_rollout_viewer_transitions = 0
        self.last_rollout_policy_transitions = 0
        self._active_demo_episode_id: Any = None
        self._active_demo_training_cycle: Any = None

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_active_demo_episode_id",
            "_active_demo_training_cycle",
            "demonstration_buffer",
            "_transition_classifier",
            "_streamer_checkpoint_hook",
        ]

    @classmethod
    def load(
        cls: type[SelfStreamerMaskablePPO],
        path: Any,
        env: Optional[Any] = None,
        device: th.device | str = "auto",
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> SelfStreamerMaskablePPO:
        """Load a stock or Streamer checkpoint and rebind sidecar state.

        SB3 applies ``kwargs`` directly to the reconstructed instance after
        ``__init__``.  Rebinding explicitly is therefore required so a loaded
        demonstration sidecar updates its retained-count diagnostic and an
        explicit BC seed initializes the sampler rather than becoming an
        otherwise-unused attribute.
        """

        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )
        demonstration_buffer = kwargs.get("demonstration_buffer")
        if demonstration_buffer is not None:
            model.set_demonstration_buffer(demonstration_buffer)
        else:
            model.set_demonstration_buffer(model.demonstration_buffer)
        model._active_demo_episode_id = None
        model._active_demo_training_cycle = None
        if kwargs.get("bc_seed") is not None:
            model.bc_seed = int(kwargs["bc_seed"])
            model._bc_rng = np.random.default_rng(model.bc_seed)
        return model

    def set_transition_classifier(
        self,
        classifier: Optional[TransitionClassifier],
    ) -> None:
        self._transition_classifier = classifier

    def set_streamer_checkpoint_hook(self, hook: Optional[Any]) -> None:
        """Install a post-optimizer-update hook for safe recovery checkpoints."""

        if hook is not None and not callable(hook):
            raise TypeError("checkpoint hook must be callable")
        self._streamer_checkpoint_hook = hook

    def set_demonstration_buffer(self, buffer: DemonstrationBuffer) -> None:
        if not isinstance(buffer, DemonstrationBuffer):
            raise TypeError("buffer must be a DemonstrationBuffer")
        if isinstance(self.observation_space, spaces.Box):
            expected_shape = tuple(int(value) for value in self.observation_space.shape)
            if buffer.observation_shape is None:
                buffer.observation_shape = expected_shape
            elif tuple(buffer.observation_shape) != expected_shape:
                raise DemonstrationValidationError(
                    "demonstration buffer observation shape does not match model: "
                    f"buffer={buffer.observation_shape} model={expected_shape}"
                )
        if isinstance(self.action_space, spaces.Discrete):
            expected_actions = int(self.action_space.n)
            if buffer.action_count is None:
                buffer.action_count = expected_actions
            elif int(buffer.action_count) != expected_actions:
                raise DemonstrationValidationError(
                    "demonstration buffer action count does not match model: "
                    f"buffer={buffer.action_count} model={expected_actions}"
                )
        self.demonstration_buffer = buffer
        self.demonstrations_recorded = len(buffer)

    def classify_transition(
        self,
        info: Mapping[str, Any],
        proposed_policy_action: int,
    ) -> BehaviorTransition:
        if self._transition_classifier is None:
            return behavior_transition_from_info(
                info,
                proposed_policy_action,
                info_key=self.transition_info_key,
            )
        classified = self._transition_classifier(info, int(proposed_policy_action))
        if isinstance(classified, BehaviorTransition):
            if classified.proposed_policy_action != int(proposed_policy_action):
                raise OffPolicyContaminationError(
                    "injected transition classifier returned a mismatched proposed action"
                )
            return classified
        if not isinstance(classified, Mapping):
            raise OffPolicyContaminationError(
                "transition classifier must return BehaviorTransition or a mapping"
            )
        return BehaviorTransition.from_mapping(
            classified,
            expected_proposed_action=int(proposed_policy_action),
        )

    @staticmethod
    def _single_observation(observation: Any) -> np.ndarray:
        if isinstance(observation, Mapping):
            raise NotImplementedError(
                "Streamer V1 demonstration storage supports the maintained flat observation only"
            )
        array = np.asarray(observation)
        if array.shape[0] != 1:
            raise ValueError(f"Streamer V1 requires one environment, got observation shape={array.shape}")
        return np.asarray(array[0], dtype=np.float32).copy()

    def _record_demonstration(
        self,
        pre_step_observation: Any,
        action_masks: Optional[np.ndarray],
        transition: BehaviorTransition,
    ) -> bool:
        if not transition.demo_eligible:
            return False
        if action_masks is None:
            if not isinstance(self.action_space, spaces.Discrete):
                raise DemonstrationValidationError(
                    "cannot synthesize a demonstration mask for a non-Discrete action space"
                )
            mask = np.ones(int(self.action_space.n), dtype=bool)
        else:
            masks = np.asarray(action_masks, dtype=bool)
            if masks.ndim != 2 or masks.shape[0] != 1:
                raise DemonstrationValidationError(
                    f"expected one batched action mask, got shape={masks.shape}"
                )
            mask = masks[0].copy()
        payload = transition.to_mapping()
        demonstration = dict(payload.get("demonstration") or {})
        demonstration.setdefault("policy_timestep", int(self.num_timesteps))
        demonstration.setdefault("environment_action", int(self.total_environment_actions))
        payload["demonstration"] = demonstration
        record = self.demonstration_buffer.add_if_eligible(
            self._single_observation(pre_step_observation),
            mask,
            payload,
        )
        if record is None:
            return False
        self.demonstrations_recorded = len(self.demonstration_buffer)
        if "episode_id" in record.metadata:
            self._active_demo_episode_id = record.metadata["episode_id"]
            self._active_demo_training_cycle = record.metadata.get("training_cycle")
        self._persist_demonstrations_if_due()
        return True

    def _persist_demonstrations_if_due(self) -> None:
        if self.demonstration_buffer.persist_path is None:
            return
        if self.demonstration_buffer.dirty_additions < self.demonstration_persist_every:
            return
        self.demonstration_buffer.save()

    def flush_demonstrations(self) -> Optional[Path]:
        if self.demonstration_buffer.persist_path is None:
            return None
        if self.demonstration_buffer.dirty_additions == 0 and self.demonstration_buffer.persist_path.exists():
            return self.demonstration_buffer.persist_path
        return self.demonstration_buffer.save()

    def _update_demonstration_outcome(
        self,
        info: Mapping[str, Any],
        done: bool,
    ) -> None:
        if not done or self._active_demo_episode_id is None:
            return
        summary = info.get("episode_summary")
        summary_mapping = summary if isinstance(summary, Mapping) else {}
        outcome = str(
            summary_mapping.get("done_reason")
            or summary_mapping.get("result")
            or info.get("done_reason")
            or info.get("terminal_reason")
            or "unknown"
        )
        self.demonstration_buffer.update_episode_outcome(
            self._active_demo_episode_id,
            outcome,
            training_cycle=self._active_demo_training_cycle,
            outcome_metadata={
                "terminal_reason": str(info.get("terminal_reason") or ""),
                "reward_total": summary_mapping.get("reward_total"),
                "win": summary_mapping.get("win"),
                "loss": summary_mapping.get("loss"),
            },
        )
        self._active_demo_episode_id = None
        self._active_demo_training_cycle = None
        self._persist_demonstrations_if_due()

    def save(
        self,
        path: str | Path,
        exclude: Optional[Sequence[str]] = None,
        include: Optional[Sequence[str]] = None,
    ) -> None:
        self.flush_demonstrations()
        super().save(path, exclude=exclude, include=include)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """Collect exactly ``n_rollout_steps`` policy-controlled transitions.

        A viewer step closes the adjacent policy segment by adding
        ``gamma * V(s_previewer)`` to its final stored reward and marking the
        first post-viewer policy sample as a fresh segment.  Stock SB3 GAE then
        computes the desired final delta while recursion cannot cross Twitch::

            delta = r_policy + gamma * V(s_previewer) - V(s_policy)
        """

        if env.num_envs != 1:
            raise NotImplementedError(
                "StreamerMaskablePPO supports exactly one environment; "
                "partial vector interventions need a valid-sample buffer"
            )
        if not isinstance(rollout_buffer, MaskableRolloutBuffer):
            raise TypeError("StreamerMaskablePPO requires MaskableRolloutBuffer with the flat Generalist observation")
        if int(n_rollout_steps) != int(rollout_buffer.buffer_size):
            raise ValueError(
                "n_rollout_steps must fill the configured rollout buffer: "
                f"requested={n_rollout_steps} buffer_size={rollout_buffer.buffer_size}"
            )
        if self._last_obs is None:
            raise RuntimeError("No previous observation was provided")
        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking")

        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        policy_steps = 0
        policy_tail_reaches_current_observation = False
        action_masks: Optional[np.ndarray] = None
        dones = np.zeros((1,), dtype=bool)
        new_obs = self._last_obs
        self.last_rollout_boundaries = []
        self.last_rollout_viewer_transitions = 0
        self.last_rollout_policy_transitions = 0

        while policy_steps < int(n_rollout_steps):
            pre_step_observation = self._last_obs
            with th.no_grad():
                obs_tensor = obs_as_tensor(pre_step_observation, self.device)
                action_masks = get_action_masks(env) if use_masking else None
                actions, values, log_probs = self.policy(
                    obs_tensor,
                    action_masks=action_masks,
                )
            actions_np = actions.cpu().numpy()
            proposed_action = int(actions_np.reshape(-1)[0])
            new_obs, rewards, dones, infos = env.step(actions_np)
            info = infos[0] if infos else {}
            transition = self.classify_transition(info, proposed_action)
            self.total_environment_actions += 1

            if transition.viewer_controlled:
                self.viewer_interventions += 1
                self.last_rollout_viewer_transitions += 1
                if policy_tail_reaches_current_observation:
                    if rollout_buffer.pos <= 0:
                        raise OffPolicyContaminationError(
                            "policy tail was marked open without an adjacent rollout sample"
                        )
                    pre_viewer_value = float(values.detach().cpu().numpy().reshape(-1)[0])
                    previous_index = int(rollout_buffer.pos - 1)
                    reward_before = float(rollout_buffer.rewards[previous_index, 0])
                    reward_after = reward_before + float(self.gamma) * pre_viewer_value
                    rollout_buffer.rewards[previous_index, 0] = reward_after
                    self.last_rollout_boundaries.append(
                        InterventionBoundary(
                            rollout_index=previous_index,
                            pre_viewer_value=pre_viewer_value,
                            gamma=float(self.gamma),
                            reward_before=reward_before,
                            reward_after=reward_after,
                        )
                    )
                policy_tail_reaches_current_observation = False
                self._record_demonstration(
                    pre_step_observation,
                    action_masks,
                    transition,
                )
                self._last_obs = new_obs
                # A viewer transition is a trajectory boundary even when the
                # underlying game episode continues.
                self._last_episode_starts = np.ones((1,), dtype=bool)
            else:
                if transition.executed_action != proposed_action:
                    raise OffPolicyContaminationError(
                        "policy behavior action changed before rollout insertion"
                    )
                self.num_timesteps += 1
                self.policy_transitions_collected += 1
                policy_steps += 1
                self.last_rollout_policy_transitions += 1

                # Preserve upstream timeout bootstrapping for actual policy
                # transitions only.  Viewer rewards never enter this path.
                for index, done in enumerate(dones):
                    if (
                        done
                        and infos[index].get("terminal_observation") is not None
                        and infos[index].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(
                            infos[index]["terminal_observation"]
                        )[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        rewards[index] += self.gamma * terminal_value

                stored_actions = actions_np
                if isinstance(self.action_space, spaces.Discrete):
                    stored_actions = stored_actions.reshape(-1, 1)
                rollout_buffer.add(
                    pre_step_observation,
                    stored_actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    action_masks=action_masks,
                )
                self._last_obs = new_obs
                self._last_episode_starts = dones
                policy_tail_reaches_current_observation = not bool(dones[0])

            self._update_info_buffer(infos, dones)
            self._update_demonstration_outcome(info, bool(dones[0]))
            callback.update_locals(locals())
            if not callback.on_step():
                return False

        if not rollout_buffer.full or rollout_buffer.pos != rollout_buffer.buffer_size:
            raise OffPolicyContaminationError(
                "rollout completed without a full policy-only buffer: "
                f"pos={rollout_buffer.pos} size={rollout_buffer.buffer_size}"
            )
        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
        callback.on_rollout_end()
        return True

    def _behavior_cloning_loss(self) -> tuple[th.Tensor, float]:
        batch = self.demonstration_buffer.sample(self.bc_batch_size, self._bc_rng)
        observations = obs_as_tensor(batch.observations, self.device)
        actions = th.as_tensor(batch.actions, dtype=th.long, device=self.device)
        action_masks = np.asarray(batch.action_masks, dtype=bool)
        distribution = self.policy.get_distribution(
            observations,
            action_masks=action_masks,
        )
        log_prob = distribution.log_prob(actions)
        if not bool(th.isfinite(log_prob).all()):
            raise DemonstrationValidationError("masked BC log probability is non-finite")
        bc_loss = -log_prob.mean()
        with th.no_grad():
            modes = distribution.get_actions(deterministic=True).long().flatten()
            agreement = float((modes == actions.flatten()).float().mean().item())
        return bc_loss, agreement

    def train(self) -> None:
        """Run stock MaskablePPO loss plus conservative masked BC."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        all_approx_kl_divs: list[float] = []
        bc_losses: list[float] = []
        bc_agreements: list[float] = []
        continue_training = True
        self._streamer_train_calls += 1
        bc_this_train = bool(
            self.bc_enabled
            and self.bc_coefficient > 0.0
            and len(self.demonstration_buffer) >= self.bc_min_demonstrations
            and (self._streamer_train_calls - 1) % self.bc_update_frequency == 0
        )
        loss = th.zeros((), device=self.device)

        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio,
                    1 - clip_range,
                    1 + clip_range,
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(float(policy_loss.item()))
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(float(clip_fraction))

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(float(value_loss.item()))

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(float(entropy_loss.item()))

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                bc_loss: Optional[th.Tensor] = None
                bc_agreement = 0.0
                if bc_this_train:
                    bc_loss, bc_agreement = self._behavior_cloning_loss()
                    # PPO's implementation minimizes its signed loss.  BC is
                    # negative log likelihood, so the coherent combined loss
                    # is +lambda * (-log pi(a_viewer | s, legal_mask)).
                    loss = loss + self.bc_coefficient * bc_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approx_kl_float = float(approx_kl_div)
                    all_approx_kl_divs.append(approx_kl_float)
                if self.target_kl is not None and approx_kl_float > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            "Early stopping due to reaching max KL: "
                            f"{approx_kl_float:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()
                if bc_loss is not None:
                    self.bc_update_count += 1
                    bc_losses.append(float(bc_loss.detach().item()))
                    bc_agreements.append(float(bc_agreement))

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.last_bc_loss = float(np.mean(bc_losses)) if bc_losses else 0.0
        self.last_bc_policy_agreement = (
            float(np.mean(bc_agreements)) if bc_agreements else 0.0
        )
        self.last_policy_loss = float(np.mean(pg_losses)) if pg_losses else 0.0
        self.last_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        self.last_entropy_loss = float(np.mean(entropy_losses)) if entropy_losses else 0.0

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record(
            "train/approx_kl",
            float(np.mean(all_approx_kl_divs)) if all_approx_kl_divs else 0.0,
        )
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", float(loss.detach().item()))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/bc_loss", self.last_bc_loss)
        self.logger.record("train/bc_coefficient", float(self.bc_coefficient))
        self.logger.record("train/bc_demonstration_count", len(self.demonstration_buffer))
        self.logger.record("train/bc_update_count", int(self.bc_update_count))
        self.logger.record(
            "train/bc_policy_agreement",
            self.last_bc_policy_agreement,
        )
        self.logger.record(
            "streamer/policy_timesteps",
            int(self.num_timesteps),
        )
        self.logger.record(
            "streamer/total_environment_actions",
            int(self.total_environment_actions),
        )
        self.logger.record(
            "streamer/viewer_interventions",
            int(self.viewer_interventions),
        )
        if self._streamer_checkpoint_hook is not None:
            self._streamer_checkpoint_hook(self)


__all__ = [
    "BEHAVIOR_SOURCE_POLICY",
    "BEHAVIOR_SOURCE_VIEWER",
    "BehaviorTransition",
    "InterventionBoundary",
    "OffPolicyContaminationError",
    "STREAMER_TRANSITION_INFO_KEY",
    "STREAMER_TRANSITION_SCHEMA_VERSION",
    "StreamerMaskablePPO",
    "TransitionClassifier",
    "behavior_transition_from_info",
]
