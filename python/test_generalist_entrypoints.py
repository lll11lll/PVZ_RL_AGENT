from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import train_ppo
from pvzrl_model_metadata import model_metadata_from_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ppo_adventure_generalist_full_v2.json"
ACTION_COUNT = 841
OBSERVATION_SHAPE = (4364,)
DECODER_VERSION = "seedslot14x60_padded6x10_plus_wait_v2"
OBSERVATION_VERSION = "adventure_14slot_identity_full_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_config(
    run_dir: Path,
    *,
    resume_model: Path | None = None,
    evaluation: bool = False,
    model_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    argv = [
        "--config",
        str(CONFIG_PATH),
        "--run-dir",
        str(run_dir),
        "--live-status-path",
        str(run_dir / "live_status.json"),
    ]
    if evaluation:
        argv.extend(
            [
                "--adventure-generalist-eval",
                "--model-path",
                str(model_path),
            ]
        )
    else:
        argv.extend(
            [
                "--adventure-generalist-train",
                "--total-timesteps",
                "8",
                "--checkpoint-freq",
                "2",
                "--no-adventure-generalist-strict-startup-validation",
            ]
        )
        if resume_model is not None:
            argv.extend(["--resume-model-path", str(resume_model)])
    args = train_ppo.build_arg_parser().parse_args(argv)
    config = train_ppo.build_config(args, train_ppo.load_json(CONFIG_PATH))
    return args, config


def _assert_generalist_contract(config: dict[str, Any]) -> None:
    metadata = train_ppo.env_metadata_for_config(config)
    assert config["run_mode"] in {
        "adventure_generalist_14slot_train",
        "adventure_generalist_14slot_eval",
    }
    assert config["action_count"] == ACTION_COUNT
    assert config["action_space_mode"] == "adventure_14slot_identity_full_v2"
    assert config["max_seed_slots"] == 14
    assert config["action_decoder_version"] == DECODER_VERSION
    assert config["observation_version"] == OBSERVATION_VERSION
    assert metadata["env_action_count"] == ACTION_COUNT
    assert tuple(metadata["observation_shape"]) == OBSERVATION_SHAPE
    assert metadata["decoder_wait_action"] == 0
    assert metadata["placement_action_range"] == [1, 840]


@pytest.fixture
def compatible_full_adventure_checkpoint(tmp_path: Path) -> Path:
    source_run = tmp_path / "full_adventure_source"
    checkpoint = source_run / "checkpoints" / "ppo_pvz_8_steps.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake-full-adventure-checkpoint\n")
    raw_config = train_ppo.load_json(CONFIG_PATH)
    (source_run / "model_metadata.json").write_text(
        json.dumps(model_metadata_from_config(raw_config), indent=2),
        encoding="utf-8",
    )
    return checkpoint


@pytest.fixture
def fake_training_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "constructor_calls": [],
        "load_calls": [],
        "learn_calls": [],
        "save_calls": [],
        "monitor_paths": [],
        "vec_envs": [],
    }

    class FakeMaskablePPO:
        def __init__(self, policy: str, env: Any, **kwargs: Any) -> None:
            self.action_space = SimpleNamespace(n=ACTION_COUNT)
            self.observation_space = SimpleNamespace(shape=OBSERVATION_SHAPE)
            self.num_timesteps = 0
            self.verbose = int(kwargs.get("verbose", 0))
            self.env = env
            state["constructor_calls"].append(
                {"policy": policy, "env": env, "kwargs": dict(kwargs), "model": self}
            )

        @classmethod
        def load(
            cls,
            path: str,
            env: Any = None,
            tensorboard_log: str | None = None,
        ) -> "FakeMaskablePPO":
            model = cls.__new__(cls)
            model.action_space = SimpleNamespace(n=ACTION_COUNT)
            model.observation_space = SimpleNamespace(shape=OBSERVATION_SHAPE)
            model.num_timesteps = 370000
            model.verbose = 0
            model.env = env
            state["load_calls"].append(
                {
                    "path": Path(path).resolve(),
                    "env": env,
                    "tensorboard_log": tensorboard_log,
                    "model": model,
                }
            )
            return model

        def learn(
            self,
            *,
            total_timesteps: int,
            callback: Any,
            reset_num_timesteps: bool,
        ) -> "FakeMaskablePPO":
            before = int(self.num_timesteps)
            if reset_num_timesteps:
                self.num_timesteps = 0
            self.num_timesteps += int(total_timesteps)
            state["learn_calls"].append(
                {
                    "model": self,
                    "total_timesteps": int(total_timesteps),
                    "callback": callback,
                    "reset_num_timesteps": bool(reset_num_timesteps),
                    "timesteps_before": before,
                    "timesteps_after": int(self.num_timesteps),
                }
            )
            return self

        def save(self, path: str) -> None:
            requested = Path(path)
            target = requested if requested.suffix == ".zip" else requested.with_suffix(".zip")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake-generalist-maskable-ppo\n")
            state["save_calls"].append(target.resolve())

    class FakeBaseCallback:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.locals: dict[str, Any] = {}

    class FakeCallbackList:
        def __init__(self, callbacks: list[Any]) -> None:
            self.callbacks = list(callbacks)

    class FakeCheckpointCallback:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)

    class FakeRuntimeEnv:
        pass

    class FakeDummyVecEnv:
        def __init__(self, factories: list[Any]) -> None:
            self.envs = [factory() for factory in factories]
            self.closed = False
            state["vec_envs"].append(self)

        def close(self) -> None:
            self.closed = True

    def fake_make_monitored_env(
        _config: dict[str, Any],
        monitor_path: Path,
        live_status_path: Path | None = None,
    ) -> FakeRuntimeEnv:
        state["monitor_paths"].append(
            {
                "monitor": Path(monitor_path).resolve(),
                "live_status": Path(live_status_path).resolve() if live_status_path else None,
            }
        )
        return FakeRuntimeEnv()

    monkeypatch.setattr(train_ppo, "require_maskable_ppo", lambda: FakeMaskablePPO)
    monkeypatch.setattr(
        train_ppo,
        "require_sb3_callbacks",
        lambda: (
            FakeBaseCallback,
            FakeCallbackList,
            FakeCheckpointCallback,
            FakeDummyVecEnv,
            object,
        ),
    )
    monkeypatch.setattr(train_ppo, "make_monitored_env", fake_make_monitored_env)
    state["model_class"] = FakeMaskablePPO
    return state


def test_fresh_generalist_training_reaches_learn_with_new_timestep_history(
    tmp_path: Path,
    fake_training_runtime: dict[str, Any],
) -> None:
    run_dir = tmp_path / "fresh_generalist"
    _args, config = _build_config(run_dir)
    _assert_generalist_contract(config)

    train_ppo.train(config, run_dir / "live_status.json")

    assert not fake_training_runtime["load_calls"]
    assert len(fake_training_runtime["constructor_calls"]) == 1
    assert len(fake_training_runtime["learn_calls"]) == 1
    learn = fake_training_runtime["learn_calls"][0]
    assert learn["reset_num_timesteps"] is True
    assert learn["timesteps_before"] == 0
    assert learn["timesteps_after"] == 8
    assert all(vec.closed for vec in fake_training_runtime["vec_envs"])

    assert (run_dir / "model.zip").is_file()
    assert (run_dir / "final_model.zip").is_file()
    assert (run_dir / "model_metadata.json").is_file()
    assert (run_dir / "resolved_config.json").is_file()
    assert (run_dir / "summary.json").is_file()

    written_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written_files
    assert all(path.is_relative_to(run_dir) for path in written_files)


def test_generalist_resume_preserves_checkpoint_and_continues_timestep_history(
    tmp_path: Path,
    fake_training_runtime: dict[str, Any],
    compatible_full_adventure_checkpoint: Path,
) -> None:
    source_hash_before = _sha256(compatible_full_adventure_checkpoint)
    source_stat_before = compatible_full_adventure_checkpoint.stat()
    run_dir = tmp_path / "resumed_generalist"
    _args, config = _build_config(run_dir, resume_model=compatible_full_adventure_checkpoint)
    _assert_generalist_contract(config)

    train_ppo.train(config, run_dir / "live_status.json")

    assert len(fake_training_runtime["constructor_calls"]) == 0
    assert len(fake_training_runtime["load_calls"]) == 1
    load = fake_training_runtime["load_calls"][0]
    assert load["path"] == compatible_full_adventure_checkpoint.resolve()
    assert load["env"] is fake_training_runtime["vec_envs"][0]
    assert Path(load["tensorboard_log"]).resolve().is_relative_to(run_dir.resolve())

    assert len(fake_training_runtime["learn_calls"]) == 1
    learn = fake_training_runtime["learn_calls"][0]
    assert learn["reset_num_timesteps"] is False
    assert learn["timesteps_before"] == 370000
    assert learn["timesteps_after"] == 370008

    assert _sha256(compatible_full_adventure_checkpoint) == source_hash_before
    source_stat_after = compatible_full_adventure_checkpoint.stat()
    assert source_stat_after.st_size == source_stat_before.st_size
    assert source_stat_after.st_mtime_ns == source_stat_before.st_mtime_ns
    assert (run_dir / "model.zip").is_file()
    assert (run_dir / "final_model.zip").is_file()
    generated = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(compatible_full_adventure_checkpoint.parents[1])
    ]
    assert all(path.is_relative_to(run_dir) for path in generated)


def test_generalist_evaluation_validates_checkpoint_before_adventure_runner(
    tmp_path: Path,
    fake_training_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    compatible_full_adventure_checkpoint: Path,
) -> None:
    run_dir = tmp_path / "generalist_eval"
    args, config = _build_config(
        run_dir,
        evaluation=True,
        model_path=compatible_full_adventure_checkpoint,
    )
    _assert_generalist_contract(config)
    calls: list[dict[str, Any]] = []

    def fake_run_adventure_eval(**kwargs: Any) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(train_ppo, "run_adventure_eval", fake_run_adventure_eval)

    train_ppo.adventure_evaluate(config, compatible_full_adventure_checkpoint, args)

    assert len(fake_training_runtime["load_calls"]) == 1
    load = fake_training_runtime["load_calls"][0]
    assert load["path"] == compatible_full_adventure_checkpoint.resolve()
    assert load["env"] is None
    assert len(calls) == 1

    call = calls[0]
    assert call["config"] is config
    assert call["model"] is load["model"]
    assert Path(call["model_path"]).resolve() == compatible_full_adventure_checkpoint.resolve()
    assert call["deterministic"] is True
    assert call["env_config"].action_space_mode == "adventure_14slot_identity_full_v2"
    assert call["env_config"].max_seed_slots == 14
    assert call["env_config"].action_decoder_version == DECODER_VERSION
    assert call["env_config"].observation_version == OBSERVATION_VERSION
    assert config["model_compatibility"]["compatible"] is True
    assert config["model_compatibility"]["model_action_count"] == ACTION_COUNT
    assert tuple(config["model_compatibility"]["model_observation_shape"]) == OBSERVATION_SHAPE
