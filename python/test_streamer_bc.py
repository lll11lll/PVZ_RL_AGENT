from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional
import zipfile

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from pvzrl_demonstrations import (
    DemonstrationBuffer,
    DemonstrationValidationError,
)
from pvzrl_streamer_ppo import StreamerMaskablePPO
import pvzrl_demonstrations as demonstrations_module


def _npy_header_only(*, shape: tuple[int, ...], dtype: np.dtype[Any]) -> bytes:
    buffer = io.BytesIO()
    buffer.write(np.lib.format.magic(1, 0))
    np.lib.format.write_array_header_1_0(
        buffer,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return buffer.getvalue()


def _rewrite_npz_member(
    source: Path,
    target: Path,
    *,
    member_name: str,
    body: bytes,
) -> None:
    with zipfile.ZipFile(source, "r") as existing, zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED
    ) as rewritten:
        for member in existing.infolist():
            rewritten.writestr(
                member.filename,
                body if member.filename == member_name else existing.read(member.filename),
            )


def _viewer_transition(
    action: int,
    *,
    eligible: bool = True,
    succeeded: bool = True,
    episode_id: str = "episode-1",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "behavior_source": "viewer",
        "viewer_controlled": True,
        "proposed_policy_action": 0,
        "executed_action": int(action),
        "execution_succeeded": bool(succeeded),
        "demo_eligible": bool(eligible),
        "execution_status": "success" if succeeded else "rejected",
        "demonstration": {
            "episode_id": episode_id,
            "observation_version": "fixture_v1",
        },
    }


def test_demonstration_buffer_accepts_only_successful_eligible_viewer_steps_and_stays_bounded(tmp_path: Path):
    path = tmp_path / "demonstrations.npz"
    buffer = DemonstrationBuffer(
        2,
        observation_shape=(2,),
        action_count=3,
        persist_path=path,
    )
    mask = np.asarray([True, True, False])

    assert buffer.add_if_eligible(
        np.asarray([0.0, 0.0]),
        mask,
        _viewer_transition(1, eligible=False),
    ) is None
    assert buffer.add_if_eligible(
        np.asarray([0.0, 0.0]),
        mask,
        {
            **_viewer_transition(1),
            "behavior_source": "policy",
            "viewer_controlled": False,
            "demo_eligible": False,
        },
    ) is None
    with pytest.raises(DemonstrationValidationError, match="execution_succeeded"):
        buffer.add_if_eligible(
            np.asarray([0.0, 0.0]),
            mask,
            _viewer_transition(1, eligible=True, succeeded=False),
        )

    buffer.add_if_eligible(
        np.asarray([1.0, 0.0]),
        mask,
        _viewer_transition(1, episode_id="episode-1"),
    )
    buffer.add_if_eligible(
        np.asarray([2.0, 0.0]),
        mask,
        _viewer_transition(0, episode_id="episode-2"),
    )
    buffer.add_if_eligible(
        np.asarray([3.0, 0.0]),
        mask,
        _viewer_transition(1, episode_id="episode-3"),
    )
    assert len(buffer) == 2
    assert buffer.total_added == 3
    assert buffer.total_evicted == 1
    np.testing.assert_allclose(
        np.stack([record.observation for record in buffer.records()]),
        [[2.0, 0.0], [3.0, 0.0]],
    )
    assert buffer.update_episode_outcome(
        "episode-3",
        "win",
        outcome_metadata={"return": 12.5},
    ) == 1
    buffer.save()

    resumed = DemonstrationBuffer.load(
        path,
        capacity=2,
        expected_observation_shape=(2,),
        expected_action_count=3,
    )
    assert len(resumed) == 2
    assert resumed.total_added == 3
    assert resumed.total_evicted == 1
    records = resumed.records()
    np.testing.assert_array_equal(records[0].action_mask, mask)
    assert records[1].metadata["episode_outcome"] == "win"
    assert records[1].metadata["episode_outcome_metadata"]["return"] == 12.5
    assert resumed.dirty_additions == 0


def test_demonstration_buffer_rejects_masked_action_and_incompatible_resume(tmp_path: Path):
    path = tmp_path / "demonstrations.npz"
    buffer = DemonstrationBuffer(
        3,
        observation_shape=(2,),
        action_count=3,
        persist_path=path,
    )
    with pytest.raises(DemonstrationValidationError, match="masked"):
        buffer.add(
            np.asarray([0.0, 0.0]),
            np.asarray([True, False, True]),
            1,
        )
    buffer.add(
        np.asarray([0.0, 0.0]),
        np.asarray([True, True, False]),
        1,
    )
    buffer.save()
    with pytest.raises(DemonstrationValidationError, match="observation shape"):
        DemonstrationBuffer.load(
            path,
            expected_observation_shape=(3,),
            expected_action_count=3,
        )
    with pytest.raises(DemonstrationValidationError, match="action count"):
        DemonstrationBuffer.load(
            path,
            expected_observation_shape=(2,),
            expected_action_count=4,
        )


def test_episode_outcome_update_is_scoped_by_training_cycle() -> None:
    buffer = DemonstrationBuffer(4, observation_shape=(2,), action_count=3)
    mask = np.asarray([True, True, False])
    for cycle in (1, 2):
        buffer.add(
            np.asarray([float(cycle), 0.0]),
            mask,
            1,
            metadata={"episode_id": 1, "training_cycle": cycle},
        )
    assert buffer.update_episode_outcome(1, "win", training_cycle=2) == 1
    records = buffer.records()
    assert "episode_outcome" not in records[0].metadata
    assert records[1].metadata["episode_outcome"] == "win"


@pytest.mark.parametrize(
    ("member_name", "malicious_header"),
    [
        (
            "metadata_json.npy",
            _npy_header_only(shape=(1,), dtype=np.dtype("<U100000")),
        ),
        (
            "format_version.npy",
            _npy_header_only(shape=(10_000_000,), dtype=np.dtype(np.int64)),
        ),
        (
            "observations.npy",
            _npy_header_only(shape=(1, 2_000_000), dtype=np.dtype(np.float32)),
        ),
    ],
)
def test_demonstration_preflight_rejects_unsafe_headers_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_name: str,
    malicious_header: bytes,
) -> None:
    source = tmp_path / "valid.npz"
    buffer = DemonstrationBuffer(
        2,
        observation_shape=(2,),
        action_count=3,
        persist_path=source,
    )
    buffer.add(
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([True, True, False]),
        1,
    )
    buffer.save()
    malicious = tmp_path / f"malicious-{member_name}"
    _rewrite_npz_member(
        source,
        malicious,
        member_name=member_name,
        body=malicious_header,
    )

    monkeypatch.setattr(
        demonstrations_module.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("np.load must not run before preflight succeeds"),
    )
    with pytest.raises(DemonstrationValidationError):
        DemonstrationBuffer.load(
            malicious,
            capacity=2,
            expected_observation_shape=(2,),
            expected_action_count=3,
        )


def test_demonstration_preflight_rejects_extra_archive_member(tmp_path: Path) -> None:
    source = tmp_path / "valid.npz"
    buffer = DemonstrationBuffer(
        1,
        observation_shape=(2,),
        action_count=3,
        persist_path=source,
    )
    buffer.save()
    malicious = tmp_path / "extra-member.npz"
    with zipfile.ZipFile(source, "r") as existing, zipfile.ZipFile(
        malicious, "w", compression=zipfile.ZIP_DEFLATED
    ) as rewritten:
        for member in existing.infolist():
            rewritten.writestr(member.filename, existing.read(member.filename))
        rewritten.writestr("unexpected.npy", b"not-an-array")
    with pytest.raises(DemonstrationValidationError, match="member set"):
        DemonstrationBuffer.load(malicious)


class TinyMaskedEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self.observation = np.asarray([0.25, -0.5], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        return self.observation.copy(), {}

    def action_masks(self) -> np.ndarray:
        return np.asarray([True, True, False], dtype=bool)

    def step(self, action: int):
        assert int(action) in {0, 1}
        return self.observation.copy(), 0.0, False, False, {}


def _masked_probabilities(model: StreamerMaskablePPO) -> np.ndarray:
    observation = th.as_tensor(
        [[0.25, -0.5]],
        dtype=th.float32,
        device=model.device,
    )
    mask = np.asarray([[True, True, False]], dtype=bool)
    with th.no_grad():
        distribution = model.policy.get_distribution(
            observation,
            action_masks=mask,
        )
        return distribution.distribution.probs.detach().cpu().numpy()[0]


def test_combined_masked_bc_uses_existing_optimizer_and_increases_demo_preference():
    env = TinyMaskedEnv()
    vec_env = DummyVecEnv([lambda: env])
    model = StreamerMaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.9,
        gae_lambda=0.8,
        learning_rate=1e-2,
        ent_coef=0.0,
        vf_coef=0.0,
        max_grad_norm=10.0,
        normalize_advantage=True,
        seed=7,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        demonstration_capacity=8,
        bc_enabled=True,
        bc_coefficient=1.0,
        bc_batch_size=1,
        bc_update_frequency=1,
        bc_min_demonstrations=1,
        bc_seed=19,
        verbose=0,
    )
    try:
        model.demonstration_buffer.add(
            np.asarray([0.25, -0.5], dtype=np.float32),
            np.asarray([True, True, False], dtype=bool),
            1,
            metadata={"episode_id": "bc-fixture"},
        )
        _total, callback = model._setup_learn(
            total_timesteps=2,
            callback=None,
            reset_num_timesteps=True,
            tb_log_name="streamer_bc_test",
            progress_bar=False,
        )
        assert model.collect_rollouts(
            vec_env,
            callback,
            model.rollout_buffer,
            n_rollout_steps=2,
            use_masking=True,
        )
        # Remove PPO actor pressure so this deterministic fixture isolates the
        # BC term inside the combined optimizer step.
        model.rollout_buffer.advantages.fill(0.0)
        model.rollout_buffer.returns[:] = model.rollout_buffer.values
        optimizer_identity = id(model.policy.optimizer)
        before = _masked_probabilities(model)
        model.train()
        after = _masked_probabilities(model)

        assert id(model.policy.optimizer) == optimizer_identity
        assert model.bc_update_count == 1
        assert np.isfinite(model.last_bc_loss)
        assert model.last_bc_loss > 0.0
        assert after[1] >= before[1]
        assert after[1] > before[1] + 1e-7
        assert after[2] == pytest.approx(0.0, abs=1e-8)
        assert model.last_bc_policy_agreement in {0.0, 1.0}
    finally:
        vec_env.close()


def test_streamer_checkpoint_resume_restores_counters_and_rebinds_demonstrations(tmp_path: Path):
    demonstration_path = tmp_path / "viewer_demonstrations.npz"
    checkpoint_path = tmp_path / "streamer_model"
    source_buffer = DemonstrationBuffer(
        4,
        observation_shape=(2,),
        action_count=3,
        persist_path=demonstration_path,
    )
    source_buffer.add(
        np.asarray([0.25, -0.5], dtype=np.float32),
        np.asarray([True, True, False], dtype=bool),
        1,
        metadata={"episode_id": "resume-fixture"},
    )
    vec_env = DummyVecEnv([TinyMaskedEnv])
    model = StreamerMaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=23,
        demonstration_buffer=source_buffer,
        bc_seed=31,
        verbose=0,
    )
    try:
        model.num_timesteps = 123
        model.total_environment_actions = 151
        model.viewer_interventions = 28
        model.policy_transitions_collected = 123
        model.bc_update_count = 7
        model._active_demo_episode_id = "must-not-survive-restart"
        model.save(checkpoint_path)

        resumed_buffer = DemonstrationBuffer.load(
            demonstration_path,
            capacity=4,
            expected_observation_shape=(2,),
            expected_action_count=3,
        )
        resumed = StreamerMaskablePPO.load(
            checkpoint_path,
            env=vec_env,
            demonstration_buffer=resumed_buffer,
            bc_seed=37,
        )
        assert resumed.num_timesteps == 123
        assert resumed.total_environment_actions == 151
        assert resumed.viewer_interventions == 28
        assert resumed.policy_transitions_collected == 123
        assert resumed.bc_update_count == 7
        assert resumed.demonstration_buffer is resumed_buffer
        assert resumed.demonstrations_recorded == 1
        assert resumed.bc_seed == 37
        assert resumed._active_demo_episode_id is None
    finally:
        vec_env.close()
